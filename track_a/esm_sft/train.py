"""Train ESM SFT scorer on AS train splits.

Usage:
    python -m track_a.esm_sft.train \\
        --encoder esm2_8m --variant base \\
        --task as_trb_i \\
        --output_dir /home/ubuntu/quest/track_a/esm_sft/results/esm2_8m-base/as_trb_i \\
        [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from track_a.esm_sft.convert_data import load_train_val
from track_a.esm_sft.dataset import TokenizedAS, collate_pad
from track_a.esm_sft.encoders import LoadedEncoder, load_encoder
from track_a.esm_sft.model import ESMScorer

WANDB_PROJECT = "neurips"
WANDB_GROUP = "esm-sft"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def evaluate_loss_auc(model, loader, device, bce):
    from sklearn.metrics import roc_auc_score
    model.eval()
    losses, preds, labels, n = [], [], [], 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = bce(logits, batch["label"]).item()
            losses.append(loss * len(batch["label"])); n += len(batch["label"])
            preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return sum(losses) / max(n, 1), float(auc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["esm2_8m", "esm2_35m", "esmc_300m"], required=True)
    p.add_argument("--variant", choices=["base", "cpt"], required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no_wandb", dest="wandb", action="store_false")
    p.set_defaults(wandb=False)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    sample_n = 100 if args.smoke else None
    epochs = 2 if args.smoke else args.epochs
    patience = 1 if args.smoke else args.patience
    batch_size = 16 if args.smoke else args.batch_size

    name = f"{args.encoder}-{args.variant}"
    print(f"[esm_sft] encoder={name} task={args.task} device={device} smoke={args.smoke}", flush=True)

    print(f"[esm_sft] loading encoder {name} ...", flush=True)
    t0 = time.time()
    loaded = load_encoder(name)
    print(f"[esm_sft]  hidden_dim={loaded.hidden_dim}  is_esmc={loaded.is_esmc}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"[esm_sft] loading data ...", flush=True)
    t0 = time.time()
    train_df, val_df = load_train_val(args.task, sample_n=sample_n, inject_negatives=True, seed=args.seed)
    print(f"[esm_sft]  train={len(train_df):,} val={len(val_df):,} ({time.time()-t0:.1f}s)", flush=True)

    chains = "paired" if args.task.startswith("as_paired") else "trb"
    mhc_col = "mhc_one_allele" if args.task.endswith("_i") else "mhc_two_allele"
    pseudoseq_col = "mhc_one_pocket_contact" if args.task.endswith("_i") else "mhc_two_pocket_contact"
    # Some rows may have missing pseudo-seq; we still pass the column name and
    # the dataset code falls back gracefully via _safe().

    train_ds = TokenizedAS(train_df, loaded.tokenizer, loaded.is_esmc, chains, mhc_col, pseudoseq_col, max_len=args.max_len)
    val_ds = TokenizedAS(val_df, loaded.tokenizer, loaded.is_esmc, chains, mhc_col, pseudoseq_col, max_len=args.max_len)
    # Choose a pad id: ESM2 has a real pad token; ESM-C uses 1 as pad too (tokenizer.pad_token_id may exist)
    pad_id = getattr(loaded.tokenizer, "pad_token_id", None) or 1

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_pad(b, pad_id=pad_id), num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            collate_fn=lambda b: collate_pad(b, pad_id=pad_id), num_workers=args.num_workers)

    model = ESMScorer(loaded, head_dim=256, dropout=0.1, freeze_base=True).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[esm_sft]  model: {n_total:,} total params, {n_train:,} trainable", flush=True)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()

    wandb_run = None; wandb_run_id = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=WANDB_PROJECT, group=WANDB_GROUP,
                name=f"{name}-{args.task}-seed{args.seed}",
                config=vars(args), reinit=True,
            )
            wandb_run_id = wandb_run.id
        except Exception as e:
            print(f"[esm_sft] wandb init failed: {e}", flush=True)

    best_val_auc = 0.0; bad = 0
    best_path = out_dir / "best.pt"
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); t0 = time.time(); loss_sum, n = 0.0, 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = bce(logits, batch["label"])
            optim.zero_grad(); loss.backward(); optim.step()
            loss_sum += loss.item() * len(batch["label"]); n += len(batch["label"])
        train_loss = loss_sum / max(n, 1)
        val_loss, val_auc = evaluate_loss_auc(model, val_loader, device, bce)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc, "secs": elapsed})
        print(f"[esm_sft]  epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  ({elapsed:.1f}s)", flush=True)
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc, "secs": elapsed})
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc; bad = 0
            torch.save({
                "encoder": args.encoder, "variant": args.variant,
                "task": args.task, "max_len": args.max_len,
                # Save head + (cpt) LoRA params, not the frozen base
                "trainable_state_dict": {k: v for k, v in model.state_dict().items()
                                         if any(t in k for t in ["pool.", "head.", "lora_"])},
            }, best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"[esm_sft]  early-stop after {epoch} epochs", flush=True)
                break

    summary = {
        "encoder": args.encoder, "variant": args.variant, "task": args.task,
        "best_val_auc": best_val_auc, "epochs_completed": len(history),
        "wandb_run_id": wandb_run_id, "history": history,
        "checkpoint": str(best_path),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[esm_sft] done. best_val_auc={best_val_auc:.4f}", flush=True)
    if wandb_run is not None: wandb_run.finish()


if __name__ == "__main__":
    main()
