#!/usr/bin/env python
# train.py
import os
import argparse
import torch.backends.cudnn as cudnn
from typing import Any
from datasets import load_from_disk  # type: ignore
# DDP training is handled by torchrun through quest.trainer.Trainer; we do not use Ray's TorchTrainer here.
from quest.trainer import Trainer


def is_main_process():
    """Returns True if the current process is the main process (rank 0)."""
    return int(os.environ.get("RANK", 0)) == 0  # `RANK` is set by torchrun


def train_model(config: dict[str, Any] | None = None, sweep: bool = False):
    cudnn.benchmark = True
    trainer = Trainer(config, sweep)
    trainer.train()


"""
This file serves as the entrypoint for LSTM/Transformer training using quest.trainer.Trainer.
Use torchrun to enable multi-GPU DDP on a single node:
  torchrun --nproc_per_node=4 scripts/ray_train.py --dataset /path/to/hf_dataset --model lstm
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to HF Dataset saved with save_to_disk')
    parser.add_argument('-m', '--model', type=str, choices=['lstm', 'bilstm', 'transformer'], default='lstm')
    parser.add_argument('-t', '--tokenizer', type=str, default=None, help='Optional tokenizer.json path; defaults to <dataset>/tokenizer.json')
    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=8, help='Transformer only')
    parser.add_argument('--dropout', type=float, default=0.1, help='Transformer only')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('-r', '--resume', action='store_true')
    parser.add_argument('-c', '--checkpoint', type=str, default='./checkpoints/lstm.pt')
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--wandb-project', type=str, default='quest')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--use-ray', action='store_true', help='Deprecated. Training is handled via torchrun/DDP if available')
    args = parser.parse_args()

    if args.use_ray:
        print("[INFO] --use-ray is deprecated in this entrypoint. Proceeding with Trainer; use torchrun for DDP.")

    os.makedirs(os.path.dirname(args.checkpoint) or '.', exist_ok=True)

    config: dict[str, Any] = {
        "dataset": args.dataset,
        "tokenizer_path": args.tokenizer,
        "model_type": args.model,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "checkpoint_path": args.checkpoint,
        "resume_from_checkpoint": args.resume,
        "early_stop": args.early_stop,
        "early_stopping_patience": args.early_stopping_patience,
        "wandb_project": args.wandb_project,
    }

    train_model(config=config, sweep=args.sweep)

if __name__ == "__main__":
    main()
