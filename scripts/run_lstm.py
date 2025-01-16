import os
import sys
import argparse
import pickle
from datetime import timedelta

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# Other libraries
import json
import pandas as pd
from tqdm import tqdm
import wandb

# Signal handling
import signal

from torch.utils.data import Dataset
import torch

class AminoAcidDataset(Dataset):
    """
    A custom PyTorch Dataset class for generating input-target pairs from amino acid sequences.

    Attributes:
        sequences (list of lists of int): A list of encoded amino acid sequences.
        seq_length (int): The desired length of each input subsequence.
        step (int): The step size for sliding the window to generate subsequences.
        pad_token (int): The token used for padding sequences (currently unused in this implementation).

    Methods:
        __len__(): Returns the total number of input-target pairs in the dataset.
        __getitem__(idx): Returns the input-target pair at the specified index.
    """

    def __init__(self, sequences, seq_length, step=1):
        """
        Initializes the AminoAcidDataset object.

        Args:
            sequences (list of lists of int): Encoded amino acid sequences.
            seq_length (int): The desired length of input subsequences.
            step (int, optional): The step size for sliding the window. Default is 1.
        """
        self.sequences = sequences
        self.seq_length = seq_length
        self.step = step

    def __len__(self):
        """
        Calculates the total number of input-target pairs in the dataset.

        Returns:
            int: The total number of input-target pairs.
        """
        # Calculate the number of subsequences for each sequence and sum them
        return sum((len(seq) - self.seq_length) // self.step for seq in self.sequences)

    def __getitem__(self, idx):
        """
        Fetches the input-target pair at the specified index.

        Args:
            idx (int): The index of the desired input-target pair.

        Returns:
            x (torch.Tensor): Input subsequence of shape (seq_length,) as a long tensor.
            y (torch.Tensor): Target token as a long tensor.
        """
        for seq in self.sequences:
            # Determine the number of valid subsequences in the current sequence
            num_subseqs = (len(seq) - self.seq_length) // self.step

            # Check if the index falls within the current sequence
            if idx < num_subseqs:
                # Calculate the start position for the subsequence
                start = idx * self.step
                # Input subsequence of length seq_length
                x = seq[start:start + self.seq_length]
                # Target is the token immediately following the subsequence
                y = seq[start + self.seq_length]
                return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
            
            # Adjust index for the next sequence
            idx -= num_subseqs

        # Raise an error if index is out of bounds (this shouldn't happen with a well-defined dataset)
        raise IndexError("Index out of range for the dataset.")

class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def signal_handler(sig, frame):
    print("Signal received: Cleaning up and exiting...")
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an LSTM model on amino acid sequences.")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Path to the training dataset ")
    return parser.parse_args()

def is_main_process():
    """Check if the current process is the main process (rank 0)."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0

# Load preprocessed dataset from file
def load_preprocessed_dataset(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def train_one_epoch(model, loader, criterion, optimizer, epoch, local_rank, world_size):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    loader.sampler.set_epoch(epoch)

    pbar = tqdm(total=len(loader), desc=f"Epoch {epoch + 1} - Training", position=0) if is_main_process() else None

    for inputs, targets in loader:
        inputs, targets = inputs.to(local_rank), targets.to(local_rank)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs, dim=1)

        # Update metrics
        total_loss += loss.item() * inputs.size(0)
        total_correct += (predicted == targets).sum().item()
        total_samples += inputs.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_main_process() and pbar is not None:
            pbar.set_postfix({"Loss": loss.item(), "Batch Accuracy": (predicted == targets).float().mean().item()})
            pbar.update(1)

    if is_main_process() and pbar is not None:
        pbar.close()

    dist.barrier()
    log_metrics(total_loss, total_correct, total_samples, "train", epoch, local_rank, world_size)


def validate_one_epoch(model, loader, criterion, epoch, local_rank, world_size):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    pbar = tqdm(total=len(loader), desc=f"Epoch {epoch + 1} - Validation", position=0) if is_main_process() else None

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(local_rank), targets.to(local_rank)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs, dim=1)

            # Update metrics
            total_loss += loss.item() * inputs.size(0)
            total_correct += (predicted == targets).sum().item()
            total_samples += inputs.size(0)

            if is_main_process() and pbar is not None:
                pbar.set_postfix({"Loss": loss.item(), "Batch Accuracy": (predicted == targets).float().mean().item()})
                pbar.update(1)

    if is_main_process() and pbar is not None:
        pbar.close()

    dist.barrier()
    log_metrics(total_loss, total_correct, total_samples, "val", epoch, local_rank, world_size)

def test_model(model, loader, criterion, local_rank, world_size):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    pbar = tqdm(total=len(loader), desc="Testing", position=0) if is_main_process() else None

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(local_rank), targets.to(local_rank)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs, dim=1)

            # Update metrics
            total_loss += loss.item() * inputs.size(0)
            total_correct += (predicted == targets).sum().item()
            total_samples += inputs.size(0)

            if is_main_process() and pbar is not None:
                pbar.set_postfix({"Loss": loss.item(), "Batch Accuracy": (predicted == targets).float().mean().item()})
                pbar.update(1)

    if is_main_process() and pbar is not None:
        pbar.close()

    dist.barrier()
    log_metrics(total_loss, total_correct, total_samples, "test", None, local_rank, world_size)

def log_metrics(total_loss, total_correct, total_samples, stage, epoch, local_rank, world_size):
    total_loss_tensor = torch.tensor(total_loss, device=local_rank)
    total_correct_tensor = torch.tensor(total_correct, device=local_rank)
    total_samples_tensor = torch.tensor(total_samples, device=local_rank)

    dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_correct_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_samples_tensor, dst=0, op=dist.ReduceOp.SUM)

    if is_main_process():
        avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
        avg_accuracy = total_correct_tensor.item() / total_samples_tensor.item()
        log_data = {f"{stage}_loss": avg_loss, f"{stage}_accuracy": avg_accuracy}
        if epoch is not None:
            log_data["epoch"] = epoch
        wandb.log(log_data)

def broadcast_config(config, local_rank):
    if dist.get_rank() == 0:
        # Serialize the wandb config on rank 0
        config_str = json.dumps(dict(config))
        config_len = torch.tensor(len(config_str), dtype=torch.int, device=f"cuda:{local_rank}")
    else:
        config_str = ""
        config_len = torch.tensor(0, dtype=torch.int, device=f"cuda:{local_rank}")

    # Broadcast the length of the config string
    dist.broadcast(config_len, src=0)

    # Create a buffer for the config string
    buffer = torch.empty(config_len.item(), dtype=torch.uint8, device=f"cuda:{local_rank}")

    if dist.get_rank() == 0:
        # Copy the string bytes into the buffer on rank 0
        buffer[:] = torch.tensor(list(config_str.encode()), dtype=torch.uint8, device=f"cuda:{local_rank}")

    # Broadcast the buffer to all ranks
    dist.broadcast(buffer, src=0)

    # Decode the string on all ranks and convert it back to a dictionary
    config_str = "".join(map(chr, buffer.cpu().tolist()))
    return json.loads(config_str)

def main():
    # --- Parse arguments and setup environment ---
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])  # Set by torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=1),
        world_size=world_size,
        rank=rank
    )

    preprocessed_data = load_preprocessed_dataset(args.dataset)

    # Set temp, and model path
    temp_path = preprocessed_data["temp_path"]

    # --- Initialize WandB (only for the main process) ---
    if is_main_process():
        config = {'seq_length': preprocessed_data["seq_length"],
              'batch_size' : preprocessed_data["batch_size"], 
              'embedding_dim': preprocessed_data["embedding_dim"],
              'hidden_dim': preprocessed_data["hidden_dim"],
              'num_layers': preprocessed_data["num_layers"],
              'learning_rate': preprocessed_data["learning_rate"],
              'num_epochs': preprocessed_data["num_epochs"],
              'source': preprocessed_data["source"],
              'percentage': preprocessed_data["percentage"]}

        wandb.init(
            project="lstm_tcr_model",
            config=config,
            name=f"{config['source']} ; {config['percentage']}%"
        )

    # Broadcast the config from rank 0 to all other ranks
    config_dict = broadcast_config(wandb.config, local_rank)
    config = argparse.Namespace(**config_dict)

    # --- Load preprocessed dataset ---
    train_dataset = preprocessed_data["train_data"]
    val_dataset = preprocessed_data["val_data"]
    test_dataset = preprocessed_data["test_data"]
    token_to_idx = preprocessed_data["token_to_idx"]
    idx_to_token = preprocessed_data["idx_to_token"]

    # --- Prepare DataLoaders ---
    batch_size = config.batch_size
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

    # --- Initialize model and optimizer ---
    vocab_size = len(token_to_idx)
    model = LSTMGenerator(
        vocab_size=vocab_size,
        embed_size=config.embedding_dim,
        hidden_size=config.hidden_dim,
        num_layers=config.num_layers
    ).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # --- Training Loop ---
    for epoch in range(config.num_epochs):
        train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, local_rank, world_size
        )

        # --- Validation Loop ---
        val_loss = validate_one_epoch(model, val_loader, criterion, epoch, local_rank, world_size)

        # Save checkpoint every 5 epochs
        if is_main_process() :
            checkpoint_path = f"{temp_path}/{config.source}_{config.percentage}_checkpoint_epoch_{epoch+1}_rank_{rank}.pth"
            torch.save(
                {
                    "model_state_dict": model.module.state_dict(),
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path
            )
            print(f"Epoch {epoch + 1}: Checkpoint saved to {checkpoint_path}")

    # --- Load the best model for testing ---
    # if is_main_process():
    #     checkpoint = torch.load(best_model_path)
    #     model.module.load_state_dict(checkpoint["model_state_dict"])
    #     print(f"Loaded best model from epoch {checkpoint['epoch']} for testing.")

    # --- Test Loop ---
    test_model(model, test_loader, criterion, local_rank, world_size)

    # --- Cleanup ---
    if is_main_process():
        wandb.finish()
    dist.destroy_process_group()


        
if __name__ == "__main__":
    main()
