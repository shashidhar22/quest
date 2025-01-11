import pandas as pd
import torch
import random
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from itertools import permutations
import argparse
import yaml
import os
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# Helper functions
def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def calculate_accuracy(outputs, targets):
    """Calculate accuracy."""
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == targets).float()
    return correct.sum() / len(correct)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an LSTM model on amino acid sequences.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file.")
    return parser.parse_args()


class AminoAcidDataset(Dataset):
    def __init__(self, sequences, seq_length, step=1):
        self.sequences = sequences
        self.seq_length = seq_length
        self.step = step

    def __len__(self):
        return sum((len(seq) - self.seq_length) // self.step for seq in self.sequences)

    def __getitem__(self, idx):
        for seq in self.sequences:
            num_subseqs = (len(seq) - self.seq_length) // self.step
            if idx < num_subseqs:
                start = idx * self.step
                x = seq[start:start + self.seq_length]
                y = seq[start + self.seq_length]
                return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
            idx -= num_subseqs


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

def is_main_process():
    """Check if the current process is the main process (rank 0)."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)

    # Initialize WandB only on the main process
    if is_main_process():
        wandb.init(project="lstm_tcr_model", config=config)

    # Initialize distributed training
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Load dataset
    dataset = pd.read_parquet(config['dataset']['path'])
    sequences = dataset['sequence'].unique().tolist()

    # Preprocess and encode data
    selected_sequences = random.sample(sequences, int(len(sequences) * config['dataset']['selection_percentage'] / 100))
    token_to_idx = {token: idx for idx, token in enumerate(set("".join(selected_sequences)))}
    encoded_sequences = [[token_to_idx[char] for char in seq] for seq in selected_sequences]

    train_size = int(0.8 * len(encoded_sequences))
    val_size = int(0.1 * len(encoded_sequences))
    test_size = len(encoded_sequences) - train_size - val_size
    train_data, val_data, test_data = random_split(encoded_sequences, [train_size, val_size, test_size])

    seq_length = config['model']['seq_length']
    batch_size = config['model']['batch_size']

    train_dataset = AminoAcidDataset(train_data, seq_length)
    val_dataset = AminoAcidDataset(val_data, seq_length)
    test_dataset = AminoAcidDataset(test_data, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=DistributedSampler(train_dataset), num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model
    vocab_size = len(token_to_idx)
    model = LSTMGenerator(vocab_size, config['model']['embed_size'], config['model']['hidden_size'], config['model']['num_layers'])
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    for epoch in range(config['training']['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        total_train_loss, total_train_accuracy = 0, 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} - Training", position=local_rank) as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(local_rank), targets.to(local_rank)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                accuracy = calculate_accuracy(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                total_train_accuracy += accuracy.item()

                pbar.set_postfix({"Loss": loss.item(), "Accuracy": accuracy.item()})
                pbar.update(1)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = total_train_accuracy / len(train_loader)
        # Log only from the main process
        if is_main_process():
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "train_accuracy": avg_train_accuracy})
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")

        # Validation loop
        model.eval()
        total_val_loss, total_val_accuracy = 0, 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} - Validation", position=local_rank) as pbar:
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(local_rank), targets.to(local_rank)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    accuracy = calculate_accuracy(outputs, targets)

                    total_val_loss += loss.item()
                    total_val_accuracy += accuracy.item()
                    pbar.set_postfix({"Loss": loss.item(), "Accuracy": accuracy.item()})
                    pbar.update(1)

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = total_val_accuracy / len(val_loader)
        # Log only from the main process
        if is_main_process():
            wandb.log({"epoch": epoch, "val_loss": avg_val_loss, "val_accuracy": avg_val_accuracy})

        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")

    # Test loop
    model.eval()
    total_test_loss, total_test_accuracy = 0, 0
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Testing", position=local_rank) as pbar:
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(local_rank), targets.to(local_rank)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                accuracy = calculate_accuracy(outputs, targets)

                total_test_loss += loss.item()
                total_test_accuracy += accuracy.item()
                pbar.set_postfix({"Loss": loss.item(), "Accuracy": accuracy.item()})
                pbar.update(1)

    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_accuracy = total_test_accuracy / len(test_loader)
            # Log only from the main process
    if is_main_process():
        wandb.log( {"test_loss": avg_test_loss, "test_accuracy": avg_test_accuracy})

    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
    # Finalize WandB on the main process
    if is_main_process():
        wandb.finish()

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        
if __name__ == "__main__":
    main()
