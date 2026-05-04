"""Evaluate trained TEIM-Seq on a TCRBench test TSV.

Loads the checkpoint written by train.py, predicts on each row of the test
TSV, computes AUC + accuracy, writes metrics JSON.

CLI:
    python evaluate.py --checkpoint smoke_run/final.ckpt --test smoke_test_iid.tsv \
        --out smoke_test_iid_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path("/home/ubuntu/quest/track_b/teim/repo")
TRAIN_DIR = REPO_DIR / "train_teim"
RESULTS_DIR = Path("/home/ubuntu/quest/track_b/teim/results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt under results/.")
    p.add_argument("--test", required=True, help="Test TSV name (under results/, no extension).")
    p.add_argument("--out", required=True, help="Output JSON name (under results/).")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    os.chdir(TRAIN_DIR)
    sys.path.insert(0, str(TRAIN_DIR))
    sys.path.insert(0, str(TRAIN_DIR / "scripts"))

    import torch
    from easydict import EasyDict
    from sklearn.metrics import accuracy_score, roc_auc_score
    from torch.utils.data import DataLoader

    from utils.dataset import SeqLevelDataset
    from train_seqlevel import SeqLevelSystem

    test_path = RESULTS_DIR / args.test
    if not test_path.exists():
        # Look for the file in repo/data/binding_data instead.
        repo_path = REPO_DIR / "data" / "binding_data" / args.test
        if repo_path.exists():
            test_path = repo_path
        else:
            raise FileNotFoundError(f"{test_path} or {repo_path}")

    # Build a SeqLevelDataset for the test split.
    test_cfg = EasyDict({
        "dataset": "seqlevel_data",
        "file_list": [test_path.stem],
        "negative": "original",
        "path": str(test_path.parent),
        "split": "",
    })
    test_set = SeqLevelDataset(test_cfg)

    # Load model from checkpoint. SeqLevelSystem doesn't call save_hyperparameters()
    # so PL's load_from_checkpoint can't reconstruct the config. We load the
    # state_dict directly into a freshly-instantiated model with the same hparams
    # the train.py used.
    ckpt_path = (RESULTS_DIR / args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    device = torch.device(args.device)
    config = EasyDict({
        "name": "eval",
        "training": {"lr": 2e-4, "epochs": 1, "batch_size": 64},
        "model": {
            "ae_model": {"dim_hid": 32, "len_epi": 12, "path": "./ckpt/epi_ae.ckpt"},
            "dim_hid": 256, "layers_inter": 2, "dim_seqlevel": 256, "inter_type": "mul",
        },
        "data": test_cfg,
    })
    model = SeqLevelSystem(config, test_set, test_set)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    loader = DataLoader(test_set, batch_size=64, shuffle=False)
    cdr3_seqs, epi_seqs, y_true, y_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            cdr3, epi, labels = batch["cdr3"].to(device), batch["epi"].to(device), batch["labels"].to(device)
            pred = model.teim_seq([cdr3, epi])["seqlevel_out"]
            cdr3_seqs.extend(batch["cdr3_seqs"])
            epi_seqs.extend(batch["epi_seqs"])
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().flatten().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pred_class = (y_pred > 0.5).astype(int)

    metrics = {
        "test": args.test,
        "n_rows": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "auc": float(roc_auc_score(y_true, y_pred)) if len(set(y_true)) > 1 else None,
        "accuracy": float(accuracy_score(y_true, pred_class)),
        "mean_score_pos": float(y_pred[y_true == 1].mean()) if (y_true == 1).any() else None,
        "mean_score_neg": float(y_pred[y_true == 0].mean()) if (y_true == 0).any() else None,
    }
    out_path = RESULTS_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
