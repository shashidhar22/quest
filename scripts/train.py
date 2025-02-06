# train.py
import os
import wandb
import argparse
import torch.backends.cudnn as cudnn

from quest.trainer import Trainer
from quest.config import SWEEP_CONFIG, DEFAULT_CONFIGS


def is_main_process():
    """Returns True if the current process is the main process (rank 0)."""
    return int(os.environ.get("RANK", 0)) == 0  # `RANK` is set by torchrun

def train_model(config=None, sweep=False):
    """
    Initializes and runs the Trainer with a given configuration.

    Args:
        config (dict, optional): Configuration dictionary containing model parameters.
    """
    cudnn.benchmark = True

    # 3) Instantiate the trainer with the normal dict
    trainer = Trainer(config, sweep)
    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-s', '--sweep', action='store_true')
    parser.add_argument('-m', '--model', type=str, choices=['lstm', 'bilstm', 'transformer'], default='lstm')
    args = parser.parse_args()

    if args.sweep:
        # Modify your SWEEP_CONFIG to set the dataset
        SWEEP_CONFIG['parameters']['dataset'] = {'value': args.dataset}
        SWEEP_CONFIG['parameters']['model_type'] = {'values': ['lstm']}
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="quest_sweep")
        wandb.agent(sweep_id, function=lambda: train_model(wandb.config, args.sweep), count=50)
    else:
        config = DEFAULT_CONFIGS[args.model].copy()
        config["dataset"] = args.dataset
        train_model(config, args.sweep)

if __name__ == "__main__":
    main()
