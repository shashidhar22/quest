"""Phase 1 cross-encoder trainer: ESM-C 300M + CPT LoRA, BCE on cube SFT corpus.

YAML-driven. Reads:
  ckpt_name, ckpt_path, tcr_variant, mhc_variant, cube_name,
  max_seq_len, output_dir,
  epochs, batch_size, lr, weight_decay, patience, neg_seed, seed,
  (eval-only fields ignored here).

Output: out_dir = <output_dir>/<ckpt_name>__<cube_name>/
  best.pt, training_summary.json, _train_done

The encoder is loaded via `track_a.esm_sft_cubes.cpt_loader.load_esmc_from_local_lora`
and wrapped in `track_a.esm_sft.model.ESMScorer` (freeze_base=True; LoRA + pool + head
are the only trainable params).

Usage:
    python -m track_a.esm_sft_cubes.train --config configs/A_cubeA.yaml [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Ensure /home/ubuntu/quest is on sys.path for `track_a.*` imports when launched
# directly (e.g. by the orchestrator subprocess).
_REPO_ROOT = "/home/ubuntu/quest"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from track_a.esm_sft.model import ESMScorer
from track_a.esm_sft_cubes.cpt_loader import load_esmc_from_local_lora
from track_a.esm_sft_cubes.cube_data import (
    cube_chains, load_train_val_with_negatives,
)
from track_a.esm_sft_cubes.dataset import TokenizedCube, collate_pad_esmc


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_loss_auc(model: nn.Module, loader: DataLoader, device: torch.device, bce: nn.Module) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    model.eval()
    losses, preds, labels, n = [], [], [], 0
    for batch in loader:
        batch = _to_device(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = bce(logits, batch["label"]).item()
        losses.append(loss * len(batch["label"])); n += len(batch["label"])
        preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(batch["label"].cpu().numpy().tolist())
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return sum(losses) / max(n, 1), float(auc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to per-run YAML")
    p.add_argument("--smoke", action="store_true", help="Tiny run (100 train rows, 1 epoch)")
    args = p.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    ckpt_name = cfg["ckpt_name"]; cube_name = cfg["cube_name"]
    output_root = Path(cfg["output_dir"])
    out_dir = output_root / f"{ckpt_name}__{cube_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist resolved config alongside the run
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if (out_dir / "_train_done").exists() and not args.smoke:
        print(f"[train] {ckpt_name}__{cube_name}: _train_done exists, skipping")
        return

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = out_dir / "train.log"
    log_f = open(log_path, "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    log(f"START {ckpt_name} × {cube_name}  device={device}  smoke={args.smoke}")

    # --- load encoder ---
    t0 = time.time()
    loaded = load_esmc_from_local_lora(cfg["ckpt_path"])
    log(f"loaded CPT adapter from {cfg['ckpt_path']}  hidden_dim={loaded.hidden_dim}  ({time.time()-t0:.1f}s)")

    # --- load data ---
    t0 = time.time()
    train_df, val_df = load_train_val_with_negatives(
        cube_name, cfg["tcr_variant"], cfg["mhc_variant"], seed=int(cfg.get("neg_seed", 43)),
    )
    if args.smoke:
        train_df = train_df.head(100); val_df = val_df.head(50)
    log(f"train n={len(train_df)}  val n={len(val_df)}  (positives+negs)  ({time.time()-t0:.1f}s)")

    chains = cube_chains(cube_name)
    train_ds = TokenizedCube(train_df, loaded.tokenizer, cfg["tcr_variant"], cfg["mhc_variant"], chains, max_len=int(cfg.get("max_seq_len", 320)))
    val_ds = TokenizedCube(val_df, loaded.tokenizer, cfg["tcr_variant"], cfg["mhc_variant"], chains, max_len=int(cfg.get("max_seq_len", 320)))
    log(f"tokenized train n={len(train_ds)}  val n={len(val_ds)}")

    pad_id = getattr(loaded.tokenizer, "pad_token_id", None) or 1
    bs = 8 if args.smoke else int(cfg.get("batch_size", 32))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=lambda b: collate_pad_esmc(b, pad_id=pad_id), num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs * 2, shuffle=False,
        collate_fn=lambda b: collate_pad_esmc(b, pad_id=pad_id), num_workers=0,
    )

    # --- model + optim ---
    model = ESMScorer(loaded, head_dim=256, dropout=0.1, freeze_base=True).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"model: {n_total/1e6:.1f}M total  {n_train/1e6:.2f}M trainable")
    peak_lr = float(cfg.get("lr", 1e-4))
    head_lr_mult = float(cfg.get("head_lr_mult", 5.0))  # pool+head LR = peak_lr × this
    # Separate param groups: pool+head are randomly initialized small modules that
    # need a higher LR to escape the constant-predictor minimum (val_auc=0.5
    # collapse seen in B, D Phase 1 runs). LoRA stays at peak_lr.
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and ("pool." in n or "head." in n)]
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]
    optim = torch.optim.AdamW([
        {"params": lora_params, "lr": peak_lr, "name": "lora"},
        {"params": head_params, "lr": peak_lr * head_lr_mult, "name": "head"},
    ], weight_decay=float(cfg.get("weight_decay", 1e-5)))
    log(f"param groups: lora={sum(p.numel() for p in lora_params)/1e6:.2f}M @ lr={peak_lr}  "
        f"head+pool={sum(p.numel() for p in head_params)/1e6:.2f}M @ lr={peak_lr * head_lr_mult}")
    bce = nn.BCEWithLogitsLoss()

    epochs = 1 if args.smoke else int(cfg.get("epochs", 3))
    patience = 1 if args.smoke else int(cfg.get("patience", 2))
    best_val_auc = 0.0; bad = 0; history = []
    best_path = out_dir / "best.pt"

    # Warmup + grad clipping: addresses catastrophic-collapse-to-prior on long-input
    # configs (D__cube_a, A__cube_c got stuck for 2-3 epochs at val_auc=0.5). 5% linear
    # warmup gives the head/LoRA gentle gradients during the first part of training.
    total_steps = epochs * len(train_loader)
    warmup_steps = int(float(cfg.get("warmup_ratio", 0.05)) * total_steps)
    grad_clip = float(cfg.get("grad_clip", 1.0))
    log(f"optim: peak_lr={peak_lr} warmup_steps={warmup_steps}/{total_steps} grad_clip={grad_clip}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train(); t0 = time.time(); loss_sum = 0.0; n = 0
        for batch in train_loader:
            # Linear warmup on lr (apply per-group scale)
            global_step += 1
            if global_step <= warmup_steps:
                scale = global_step / max(warmup_steps, 1)
                for g in optim.param_groups:
                    base = peak_lr if g.get("name") == "lora" else peak_lr * head_lr_mult
                    g["lr"] = base * scale
            batch = _to_device(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = bce(logits, batch["label"])
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            optim.step()
            loss_sum += loss.item() * len(batch["label"]); n += len(batch["label"])
        train_loss = loss_sum / max(n, 1)
        val_loss, val_auc = evaluate_loss_auc(model, val_loader, device, bce)
        elapsed = time.time() - t0
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc, "secs": elapsed}
        history.append(rec)
        log(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  ({elapsed:.1f}s)")
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc; bad = 0
            torch.save({
                "ckpt_name": ckpt_name, "cube_name": cube_name,
                "cfg": cfg, "best_val_auc": best_val_auc,
                "trainable_state_dict": {k: v for k, v in model.state_dict().items()
                                         if any(t in k for t in ["pool.", "head.", "lora_"])},
            }, best_path)
        else:
            bad += 1
            if bad >= patience:
                log(f"early-stop after epoch {epoch}"); break

    summary = {
        "ckpt_name": ckpt_name, "cube_name": cube_name,
        "tcr_variant": cfg["tcr_variant"], "mhc_variant": cfg["mhc_variant"],
        "best_val_auc": best_val_auc, "epochs_completed": len(history),
        "history": history, "checkpoint": str(best_path),
        "smoke": args.smoke,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    if not args.smoke:
        (out_dir / "_train_done").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    log(f"DONE  best_val_auc={best_val_auc:.4f}")
    log_f.close()


if __name__ == "__main__":
    main()
