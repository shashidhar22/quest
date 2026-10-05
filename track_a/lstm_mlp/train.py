"""Training entry point for LSTM/MLP AS retrieval baselines.

Usage:
    python -m track_a.lstm_mlp.train \\
        --model lstm \\
        --task as_trb_i \\
        --output_dir /home/ubuntu/quest/track_a/lstm_mlp/results/lstm/as_trb_i \\
        [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from track_a.lstm_mlp.convert_data import load_train_val
from track_a.lstm_mlp.dataset import (
    DEFAULT_MAX_PEP,
    DEFAULT_MAX_TRA,
    DEFAULT_MAX_TRB,
    TripletDataset,
    build_allele_vocab,
    collate,
)
from track_a.lstm_mlp.model import TCRPeptideMHCScorer

WANDB_PROJECT = "neurips"
WANDB_GROUP = "baselines"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _forward(model: TCRPeptideMHCScorer, batch: dict) -> torch.Tensor:
    if model.chains == "paired":
        return model(
            trb_ids=batch["trb_ids"], trb_len=batch["trb_len"],
            pep_ids=batch["pep_ids"], pep_len=batch["pep_len"],
            mhc_oh=batch["mhc_oh"],
            tra_ids=batch["tra_ids"], tra_len=batch["tra_len"],
        )
    return model(
        trb_ids=batch["trb_ids"], trb_len=batch["trb_len"],
        pep_ids=batch["pep_ids"], pep_len=batch["pep_len"],
        mhc_oh=batch["mhc_oh"],
    )


def evaluate_loss_auc(model: TCRPeptideMHCScorer, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    model.eval()
    losses, preds, labels = [], [], []
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            logits = _forward(model, batch)
            loss = bce(logits, batch["label"]).item()
            losses.append(loss)
            n += len(batch["label"])
            preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return sum(losses) / max(n, 1), float(auc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lstm", "mlp"], required=True)
    p.add_argument("--task", required=True, help="e.g. as_trb_i, as_paired_i, as_trb_ii, as_paired_ii")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--smoke", action="store_true", help="100/50/50 row smoke test, 2 epochs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no_wandb", dest="wandb", action="store_false")
    p.set_defaults(wandb=False)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Smoke-test overrides
    sample_n = 100 if args.smoke else None
    epochs = 2 if args.smoke else args.epochs
    patience = 1 if args.smoke else args.patience

    print(f"[lstm_mlp] task={args.task} model={args.model} device={device} smoke={args.smoke}", flush=True)

    print("[lstm_mlp] loading train+val splits + sampling negatives ...", flush=True)
    t0 = time.time()
    train_df, val_df = load_train_val(args.task, sample_n=sample_n, inject_negatives=True, seed=args.seed)
    print(f"[lstm_mlp]  train={len(train_df):,} val={len(val_df):,} ({time.time()-t0:.1f}s)", flush=True)

    mhc_col = "mhc_one_allele" if args.task.endswith("_i") else "mhc_two_allele"
    allele_vocab = build_allele_vocab(train_df, mhc_col=mhc_col, cache_path=out_dir / "allele_vocab.json")
    mhc_dim = len(allele_vocab)
    print(f"[lstm_mlp]  allele vocab size={mhc_dim}", flush=True)

    train_ds = TripletDataset(train_df, args.task, allele_vocab)
    val_ds = TripletDataset(val_df, args.task, allele_vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    chains = "paired" if args.task.startswith("as_paired") else "trb"
    model = TCRPeptideMHCScorer(encoder_type=args.model, chains=chains, mhc_dim=mhc_dim).to(device)
    print(f"[lstm_mlp]  model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()

    wandb_run = None
    wandb_run_id = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=WANDB_PROJECT, group=WANDB_GROUP,
                name=f"{args.model}-{args.task}-seed{args.seed}",
                config=vars(args), reinit=True,
            )
            wandb_run_id = wandb_run.id
        except Exception as e:
            print(f"[lstm_mlp] wandb init failed: {e}; continuing without", flush=True)

    best_val_auc = 0.0
    bad_epochs = 0
    best_path = out_dir / "best.pt"
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, n = 0.0, 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            logits = _forward(model, batch)
            loss = bce(logits, batch["label"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += loss.item() * len(batch["label"])
            n += len(batch["label"])
        train_loss = loss_sum / max(n, 1)
        val_loss, val_auc = evaluate_loss_auc(model, val_loader, device)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc, "secs": elapsed})
        print(f"[lstm_mlp]  epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  ({elapsed:.1f}s)", flush=True)
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc, "secs": elapsed})
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            bad_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "encoder_type": args.model,
                "chains": chains,
                "mhc_dim": mhc_dim,
                "allele_vocab": allele_vocab,
                "task": args.task,
                "max_trb": DEFAULT_MAX_TRB,
                "max_tra": DEFAULT_MAX_TRA,
                "max_pep": DEFAULT_MAX_PEP,
            }, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[lstm_mlp]  early-stop after {epoch} epochs (no improvement for {patience})", flush=True)
                break

    summary = {
        "model": args.model,
        "task": args.task,
        "best_val_auc": best_val_auc,
        "epochs_completed": len(history),
        "wandb_run_id": wandb_run_id,
        "history": history,
        "checkpoint": str(best_path),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[lstm_mlp] done. best_val_auc={best_val_auc:.4f}  ckpt={best_path}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
