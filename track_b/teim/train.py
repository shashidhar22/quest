"""Train TEIM-Seq on TCRBench AS data.

Imports `SeqLevelDataset` and `SeqLevelSystem` from the upstream repo and
runs PL training with our own config / data pipeline. We do this rather than
calling `train_teim/scripts/train_seqlevel.py` directly because the upstream
script hard-codes `gpus=1` (would crash on CPU), reads the config YAML in a
specific layout, and writes outputs to a relative `logs/` dir.

CLI:
    python train.py --train smoke_train --val smoke_val --epochs 2 --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_DIR = Path("/home/ubuntu/quest/track_b/teim/repo")
TRAIN_DIR = REPO_DIR / "train_teim"
RESULTS_DIR = Path("/home/ubuntu/quest/track_b/teim/results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Train TSV name (under repo/data/binding_data/, no extension).")
    p.add_argument("--val", required=True, help="Val TSV name (under repo/data/binding_data/, no extension).")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--gpu-id", type=int, default=2)
    p.add_argument("--name", default="teim_seq_smoke")
    args = p.parse_args()

    # Run from train_teim/ so relative paths in upstream code resolve.
    os.chdir(TRAIN_DIR)
    sys.path.insert(0, str(TRAIN_DIR))
    sys.path.insert(0, str(TRAIN_DIR / "scripts"))

    from easydict import EasyDict
    from torch.utils.data import Subset
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    from utils.dataset import SeqLevelDataset
    from train_seqlevel import SeqLevelSystem

    pl.seed_everything(0)

    # Point at our results dir (NOT repo/data/binding_data) so the upstream
    # SeqLevelDataset doesn't try to read positive_epi_dist.tsv (which only
    # contains TEIM's training peptides; our peptides aren't there → KeyError).
    train_cfg = EasyDict({
        "dataset": "seqlevel_data",
        "file_list": [args.train],
        "negative": "original",  # use our pre-labeled rows as-is
        "path": str(RESULTS_DIR),
        "split": "",  # disable upstream's split logic; we pass train_set/val_set explicitly
    })
    val_cfg = EasyDict({**train_cfg, "file_list": [args.val]})

    train_set = SeqLevelDataset(train_cfg)
    val_set = SeqLevelDataset(val_cfg)

    config = EasyDict({
        "name": args.name,
        "training": {"lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size},
        "model": {
            "ae_model": {"dim_hid": 32, "len_epi": 12, "path": "./ckpt/epi_ae.ckpt"},
            "dim_hid": 256, "layers_inter": 2, "dim_seqlevel": 256, "inter_type": "mul",
        },
        "data": train_cfg,
    })

    model = SeqLevelSystem(config, train_set, val_set)
    out_dir = RESULTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(monitor="valid/auc_avg", save_last=True, mode="max", save_top_k=1),
        EarlyStopping(monitor="valid/auc_avg", patience=15, mode="max"),
    ]
    trainer_kwargs = dict(
        max_epochs=args.epochs,
        callbacks=callbacks,
        default_root_dir=str(out_dir),
        # Disable progress bar noise on CPU.
        enable_progress_bar=True,
    )
    if args.device == "cuda":
        trainer_kwargs["gpus"] = [args.gpu_id]
    else:
        trainer_kwargs["gpus"] = 0  # CPU
    trainer = pl.Trainer(**trainer_kwargs)
    print(f"Training on {args.device}, {len(train_set)} train rows, {len(val_set)} val rows.")
    trainer.fit(model)

    # Save the final state_dict for evaluate.py.
    import torch
    final_ckpt = out_dir / "final.ckpt"
    trainer.save_checkpoint(str(final_ckpt))
    print(f"Saved checkpoint to {final_ckpt}")


if __name__ == "__main__":
    main()
