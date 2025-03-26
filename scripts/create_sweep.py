#!/usr/bin/env python

import wandb
import argparse
from quest.config import SWEEP_CONFIG, DEFAULT_CONFIGS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, choices=['lstm', 'bilstm', 'transformer', 'bert'], default='lstm')
    parser.add_argument('-t', '--tokenizer', type=str, default=None)
    args = parser.parse_args()

    if args.tokenizer:
            tokenizer_path = args.tokenizer
    else:
        tokenizer_path = None
        
    # Modify your SWEEP_CONFIG to set the dataset
    percentages = SWEEP_CONFIG['parameters']['data_size']['values']
    slen = SWEEP_CONFIG['parameters']['sequence_length']['values']
    vocab = SWEEP_CONFIG['parameters']['vocab_size']['values']

    datasets = [ f"{args.dataset}_{p}P_{s}L_{v}V" for p in percentages for s in slen for v in vocab]
    SWEEP_CONFIG['parameters']['dataset'] = {'value': datasets}
    SWEEP_CONFIG['parameters']['tokenizer_path'] = {'value': tokenizer_path}
    SWEEP_CONFIG['parameters']['method'] = {'value': 'bayes'}

    sweep_id = wandb.sweep(SWEEP_CONFIG, project=f"quest_{args.model}_sweep")
    print(f"Sweep ID: {sweep_id}")
