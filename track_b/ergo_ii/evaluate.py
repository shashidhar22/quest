"""Evaluate a trained ERGO-II model on a TCRBench AS-TRB-I test split.

Loads the checkpoint written by train.py, scores rows from a
`<dataset_name>_<split>_samples.pickle`, and reports AUC + accuracy.

CLI:
    python evaluate.py --dataset-name smoke --split test_iid --train-dataset-name smoke
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from argparse import Namespace
from pathlib import Path

REPO_DIR = Path("/home/ubuntu/quest/track_b/ergo_ii/repo")
RESULTS_DIR = Path("/home/ubuntu/quest/track_b/ergo_ii/results")
SAMPLES_DIR = REPO_DIR / "Samples"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True,
                        help="Eval-split pickle prefix (matches convert_data.py output).")
    parser.add_argument("--split", required=True,
                        help="Eval split label (e.g. test_iid, test_novel_tcr).")
    parser.add_argument("--train-dataset-name", required=True,
                        help="The dataset_name used during training. Determines the V/J/MHC vocab.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu-id", type=int, default=2)
    args = parser.parse_args()

    os.chdir(REPO_DIR)
    sys.path.insert(0, str(REPO_DIR))

    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score

    from Trainer import ERGOLightning
    from Loader import SignedPairsDataset, get_index_dicts
    from torch.utils.data import DataLoader

    ckpt_path = RESULTS_DIR / "checkpoints" / f"{args.train_dataset_name}_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} — run train.py first.")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = Namespace(**ckpt["hparams"])
    # Override the training-only `dataset` field so the model loads the
    # train pickle for vocab construction (V/J/MHC index dicts).
    hparams.dataset = args.train_dataset_name

    model = ERGOLightning(hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if args.device == "cuda":
        model = model.cuda(args.gpu_id)

    # Build vocab from train pickle.
    train_pkl = SAMPLES_DIR / f"{args.train_dataset_name}_train_samples.pickle"
    with open(train_pkl, "rb") as f:
        train_samples = pickle.load(f)
    index_dicts = get_index_dicts(train_samples)

    # Load eval pickle.
    eval_pkl = SAMPLES_DIR / f"{args.dataset_name}_{args.split}_samples.pickle"
    with open(eval_pkl, "rb") as f:
        eval_samples = pickle.load(f)
    eval_dataset = SignedPairsDataset(eval_samples, index_dicts)
    loader = DataLoader(
        eval_dataset, batch_size=64, shuffle=False, num_workers=0,
        collate_fn=lambda b: eval_dataset.collate(
            b, tcr_encoding=model.tcr_encoding_model, cat_encoding=model.cat_encoding,
        ),
    )

    y_all, y_hat_all = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            if args.device == "cuda":
                batch = [t.cuda(args.gpu_id) if torch.is_tensor(t) else t for t in batch]
            y, y_hat, _ = model.step(batch)
            y_all.append(y.detach().cpu().numpy())
            y_hat_all.append(y_hat.detach().cpu().numpy())
    y = np.concatenate(y_all)
    y_hat = np.concatenate(y_hat_all)
    pred = (y_hat > 0.5).astype(int)

    metrics = {
        "split": args.split,
        "n_rows": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "auc": float(roc_auc_score(y, y_hat)) if len(set(y)) > 1 else None,
        "accuracy": float(accuracy_score(y, pred)),
        "mean_score_pos": float(y_hat[y == 1].mean()) if (y == 1).any() else None,
        "mean_score_neg": float(y_hat[y == 0].mean()) if (y == 0).any() else None,
    }

    out_path = RESULTS_DIR / f"{args.dataset_name}_{args.split}_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
