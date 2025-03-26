#!/usr/bin/env python
# train.py
import os
import wandb
import argparse
import torch.backends.cudnn as cudnn

from quest.trainer import Trainer
from quest.config import SWEEP_CONFIG, DEFAULT_CONFIGS
from quest.models.bert import train_bert_with_transformers

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
    parser.add_argument('-s', '--sweep', type='store_true')
    parser.add_argument('-i', '--sweepid', type=int, default=0)
    parser.add_argument('-m', '--model', type=str, choices=['lstm', 'bilstm', 'transformer', 'bert'], default='lstm')
    parser.add_argument('-l', '--log', type=str, default='temp/logs')
    parser.add_argument('-r', '--resume', action='store_true')
    parser.add_argument('-c', '--checkpoint', type=str, default=None)
    parser.add_argument('-t', '--tokenizer', type=str, default=None)
    args = parser.parse_args()

    if args.model == "bert":
        # 1) Decide tokenizer_path
        if args.tokenizer:
            tokenizer_path = args.tokenizer
        else:
            tokenizer_path = f'{args.dataset}/tokenizer.json'

        # 2) Decide checkpoint_path
        if not args.checkpoint:
            checkpoint_path = f'data/models/bert.pt'
        else:
            checkpoint_path = args.checkpoint
        # => Call a new function that runs a Hugging Face Trainer
        if args.sweep:
            # Modify your SWEEP_CONFIG to set the dataset
            SWEEP_CONFIG['parameters']['dataset'] = {'value': args.dataset}
            SWEEP_CONFIG['parameters']['tokenizer_path'] = {'value': tokenizer_path}
            SWEEP_CONFIG['parameters']['method'] = {'value': 'bayes'}
            sweep_id = wandb.sweep(SWEEP_CONFIG, project="quest_sweep")

            # This function needs to pass a DICT to train_bert_with_transformers
            def sweep_train_bert():
                sweep_cfg = dict(wandb.config)  # or wandb.config itself
                train_bert_with_transformers(sweep_cfg)

            wandb.agent(sweep_id, function=sweep_train_bert, count=50)
        else:
            # => Single-run scenario. Build config from your defaults:
            config = DEFAULT_CONFIGS["bert"].copy()
            # e.g. config could have "num_epochs", "batch_size", etc.
            config["checkpoint_path"] = checkpoint_path
            config["dataset"] = args.dataset
            config["tokenizer_path"] = tokenizer_path
            # If "resume_training" is relevant, set that too:
            config["resume_training"] = args.resume

            # Finally call your HF training function
            train_bert_with_transformers(config)
    else:
        if args.tokenizer:
            tokenizer_path = args.tokenizer
        else:
            tokenizer_path = None
        if not args.checkpoint:
            checkpoint_path = f'data/models/{args.model}.pt'
        else:
            checkpoint_path = args.checkpoint
        if args.sweep:
            # Modify your SWEEP_CONFIG to set the dataset
            percentages = SWEEP_CONFIG['parameters']['data_size']['values']
            slen = SWEEP_CONFIG['parameters']['sequence_length']['values']
            vocab = SWEEP_CONFIG['parameters']['vocab_size']['values']

            datasets = [ f"{args.dataset}_{p}P_{s}L_{v}V" for p in percentages for s in slen for v in vocab]
            SWEEP_CONFIG['parameters']['dataset'] = {'value': datasets}
            SWEEP_CONFIG['parameters']['tokenizer_path'] = {'value': tokenizer_path}
            SWEEP_CONFIG['parameters']['method'] = {'value': 'bayes'}
            if args.sweepid:
                sweep_id = args.sweepid
            else:
                sweep_id = wandb.sweep(SWEEP_CONFIG, project="quest_lstm_sweep")
            wandb.agent(sweep_id, function=lambda: train_model(wandb.config, args.sweep), count=10)
        else:
            config = DEFAULT_CONFIGS[args.model].copy()
            config['checkpoint_path'] = checkpoint_path
            config['resume_training'] = args.resume
            config['dataset'] = args.dataset
            config['tokenizer_path'] = {'value': tokenizer_path}
            train_model(config, args.sweep)

if __name__ == "__main__":
    main()
