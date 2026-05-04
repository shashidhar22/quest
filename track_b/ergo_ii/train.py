"""Train ERGO-II on TCRBench data.

Wraps repo/Trainer.py. ERGO-II's train_dataloader hard-codes
`Samples/{dataset}_train_samples.pickle` and val_dataloader hard-codes
`Samples/{dataset}_test_samples.pickle`. This wrapper:

1. Verifies the two expected pickles exist (build them with convert_data.py).
2. Builds an hparams Namespace with our chosen flags (LSTM + vj + mhc + t_type).
3. Calls Trainer.fit on a CPU-only or GPU lightning Trainer.
4. Saves the final checkpoint to track_b/ergo_ii/results/checkpoints/.

CLI:
    python train.py --dataset-name smoke --epochs 2 --device cpu
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from argparse import Namespace
from pathlib import Path

REPO_DIR = Path("/home/ubuntu/quest/track_b/ergo_ii/repo")
RESULTS_DIR = Path("/home/ubuntu/quest/track_b/ergo_ii/results")
SAMPLES_DIR = REPO_DIR / "Samples"


def make_hparams(dataset_name: str, epochs: int) -> Namespace:
    return Namespace(
        # ERGO-II.vdj LSTM mode
        dataset=dataset_name,
        tcr_encoding_model="LSTM",
        cat_encoding="embedding",
        use_alpha=False,    # AS-TRB-I has no TRA
        use_vj=True,
        use_mhc=True,
        use_t_type=False,   # TCRBench has no CD4/CD8 annotation
        # Hyperparameters from paper §"Configurations and Hyperparameters Tuning"
        aa_embedding_dim=10,
        cat_embedding_dim=50,
        lstm_dim=500,
        encoding_dim=100,
        lr=1e-4,
        wd=1e-5,
        dropout=0.1,
        # Smoke-test override
        max_epochs=epochs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True,
                        help="Must match the prefix used by convert_data.py.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="cpu is required on L4 (sm_89 incompat with torch 1.7).")
    parser.add_argument("--gpu-id", type=int, default=2, help="Only used if --device cuda.")
    args = parser.parse_args()

    # Verify pickles exist.
    train_pkl = SAMPLES_DIR / f"{args.dataset_name}_train_samples.pickle"
    test_pkl = SAMPLES_DIR / f"{args.dataset_name}_test_samples.pickle"
    if not train_pkl.exists():
        raise FileNotFoundError(f"{train_pkl} — run convert_data.py --split train first")
    if not test_pkl.exists():
        raise FileNotFoundError(
            f"{test_pkl} — ERGO-II's val_dataloader needs a {args.dataset_name}_test_samples.pickle. "
            "Run convert_data.py --split val (write a small held-out chunk under that filename)."
        )

    # ERGO-II uses relative paths, so we cwd into the repo.
    os.chdir(REPO_DIR)
    sys.path.insert(0, str(REPO_DIR))

    from Trainer import ERGOLightning
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.logging import TensorBoardLogger

    hparams = make_hparams(args.dataset_name, args.epochs)
    model = ERGOLightning(hparams)
    logger = TensorBoardLogger(str(RESULTS_DIR / "tb_logs"), name=args.dataset_name)
    early_stop = EarlyStopping(monitor="val_auc", patience=3, mode="max")

    trainer_kwargs = dict(
        logger=logger,
        early_stop_callback=early_stop,
        max_epochs=args.epochs,
    )
    if args.device == "cuda":
        trainer_kwargs["gpus"] = [args.gpu_id]
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)

    # Save the final state_dict alongside the lightning logger output for
    # easy reload from evaluate.py.
    ckpt_dir = RESULTS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(
        {"state_dict": model.state_dict(), "hparams": vars(hparams)},
        ckpt_dir / f"{args.dataset_name}_final.pt",
    )
    print(f"Saved checkpoint to {ckpt_dir / f'{args.dataset_name}_final.pt'}")


if __name__ == "__main__":
    main()
