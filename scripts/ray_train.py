#!/usr/bin/env python
# train.py
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import wandb
from typing import Any
from tqdm import tqdm
from datasets import load_from_disk  # type: ignore
from tokenizers import Tokenizer, models, trainers, pre_tokenizers  # type: ignore
from torch.utils.data import DataLoader
from ray.train.torch import TorchTrainer  # type: ignore
from ray.train import ScalingConfig, RunConfig, FailureConfig, CheckpointConfig  # type: ignore
from ray.air import session  # type: ignore
import ray
from quest.trainer import Trainer
from quest.models.lstm import LSTMModel  # type: ignore
from quest.models.transformer import TransformerModel  # type: ignore
from quest.dataset import AminoAcidDataset  # type: ignore


def is_main_process():
    """Returns True if the current process is the main process (rank 0)."""
    return int(os.environ.get("RANK", 0)) == 0  # `RANK` is set by torchrun


def train_model(config: dict[str, Any] | None = None, sweep: bool = False):
    cudnn.benchmark = True
    trainer = Trainer(config, sweep)
    trainer.train()


def ray_train_func(config: dict[str, Any]) -> None:
    # 1. Load dataset
    ds = load_from_disk(config["dataset_path"])
    train_split = ds["train"] if isinstance(ds, dict) or hasattr(ds, "__getitem__") else ds  # type: ignore
    val_split = ds["val"] if "val" in ds else None  # type: ignore

    # 2. Load or train BPE tokenizer
    tokenizer_path = config.get("tokenizer_path", "tokenizer.json")
    vocab_size = config.get("bpe_vocab", 200)
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)  # type: ignore
    else:
        def tag_bpe(row: dict[str, Any]):
            return " ".join(list(row["combo_id"]))
        train_seqs = (tag_bpe(row) for row in train_split)  # type: ignore
        tokenizer = Tokenizer(models.BPE())  # type: ignore
        tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=r" ", behavior="removed")  # type: ignore
        special_tokens = ["[PAD]", "[UNK]", "[END]", "[TRA]", "[ETRA]", "[TRB]", "[ETRB]", "[PEP]", "[EPEP]", "[MHO]", "[EMHO]", "[MHT]", "[EMHT]"]
        trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)  # type: ignore
        tokenizer.train_from_iterator(train_seqs, trainer)  # type: ignore
        tokenizer.save(tokenizer_path)  # type: ignore

    seq_len = config.get("seq_len", 128)
    batch_size = config.get("batch_size", 32)
    num_epochs = config.get("num_epochs", 3)
    model_type = config.get("model_type", "lstm")

    # Prepare raw sequences for AminoAcidDataset
    def get_sequences(hf_dataset: Any) -> list[str]:
        # Each row may have multiple combo_ids (list), flatten all
        seqs = []
        for row in hf_dataset:
            combo_ids = row["combo_id"]
            if isinstance(combo_ids, list):
                seqs.extend(combo_ids)  # type: ignore
            else:
                seqs.append(combo_ids)  # type: ignore
        return seqs  # type: ignore

    train_sequences = get_sequences(train_split)
    val_sequences = get_sequences(val_split) if val_split is not None else None

    # Use AminoAcidDataset for both LSTM and Transformer
    train_dataset = AminoAcidDataset(
        train_sequences,
        tokenizer,  # type: ignore
        seq_length=seq_len,
        model_type="rnn" if model_type == "lstm" else "transformer"
    )  # type: ignore
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)  # type: ignore
    val_loader = None
    if val_sequences is not None:
        val_dataset = AminoAcidDataset(
            val_sequences,
            tokenizer,  # type: ignore
            seq_length=seq_len,
            model_type="rnn" if model_type == "lstm" else "transformer"
        )  # type: ignore
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "lstm":
        model = LSTMModel(**config["model_args"]).to(device)  # type: ignore
    elif model_type == "transformer":
        model = TransformerModel(**config["model_args"]).to(device)  # type: ignore
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))  # type: ignore
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))  # type: ignore

    # W&B logging (only on rank 0)
    wandb_project = config.get("wandb_project")
    if wandb_project and (not hasattr(ray, "train") or session.get_world_rank() == 0):  # type: ignore
        wandb.init(project=wandb_project, config=config)

    for epoch in range(num_epochs):
        model.train()  # type: ignore
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):  # type: ignore
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)  # type: ignore
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs  # type: ignore
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))  # type: ignore
            loss.backward()
            optimizer.step()  # type: ignore
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)  # type: ignore
        val_loss = None
        if val_loader is not None:
            model.eval()  # type: ignore
            val_total_loss = 0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"Validation {epoch+1}"):  # type: ignore
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)  # type: ignore
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs  # type: ignore
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))  # type: ignore
                    val_total_loss += loss.item()
            val_loss = val_total_loss / len(val_loader)  # type: ignore
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f} val loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        # Log to wandb if initialized
        if wandb.run:
            wandb.log({"train_loss": avg_loss, "val_loss": val_loss, "epoch": epoch+1})
        session.report({"epoch": epoch+1, "avg_loss": avg_loss, "val_loss": val_loss})  # type: ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, choices=['lstm', 'bilstm', 'transformer'], default='lstm')
    parser.add_argument('-l', '--log', type=str, default='temp/logs')
    parser.add_argument('-r', '--resume', action='store_true')
    parser.add_argument('-c', '--checkpoint', type=str, default=None)
    parser.add_argument('-t', '--tokenizer', type=str, default=None)
    parser.add_argument('--use-ray', action='store_true', help='Use Ray for distributed training')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of Ray workers (GPUs)')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    if args.use_ray:
        if not ray.is_initialized():  # type: ignore
            ray.init()  # type: ignore
        trainer = TorchTrainer(
            ray_train_func,
            scaling_config=ScalingConfig(
                num_workers=args.num_workers,
                use_gpu=True,
            ),
            run_config=RunConfig(
                failure_config=FailureConfig(max_failures=-1),
                checkpoint_config=CheckpointConfig(num_to_keep=1),
            ),
            train_loop_config={
                # Pass all config needed for ray_train_func
                # e.g. dataset path, model args, etc.
            }
        )
        result = trainer.fit()
        print("Ray DDP training finished!", result)
        return

    # Local/single-node training logic for LSTM/Transformer can be added here if needed

if __name__ == "__main__":
    main()
