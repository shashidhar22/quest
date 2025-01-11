import pandas as pd
import time
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from itertools import permutations
import argparse
import yaml
import wandb

# Function to load configuration from a YAML file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to set up command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM model on amino acid sequences.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file.")
    return parser.parse_args()

# Main function to run the training process
def main():
    # Parse arguments
    args = parse_args()
    
    # Load config from YAML
    config = load_config(args.config)


    # Initialize WandB for tracking
    wandb.init(project="lstm_tcr_model", config=config)

     # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.cuda.empty_cache()

    # Load dataset
    dataset = pd.read_parquet(config['dataset']['path'])
    sequences = dataset['sequence'].unique().tolist()

    # Generate combinations and individual molecules from the sequence
    def generate_combinations_and_individuals(seq):
        if not seq.endswith(";"):
            seq += ";"
        molecules = seq.split()
        result = [molecule + ";" for molecule in molecules]
        for r in range(2, len(molecules) + 1):
            result.extend([" ".join(perm) + ";" for perm in permutations(molecules, r)])
        return result

    expanded_sequences = []
    for seq in sequences:
        if ' ' in seq:
            expanded_sequences.extend(generate_combinations_and_individuals(seq))
        else:
            if not seq.endswith(";"):
                seq += ";"
            expanded_sequences.append(seq)

    expanded_sequences = list(set(expanded_sequences))
    print(f"Original dataset had {len(sequences)} unique sequences.")
    print(f"Expanded dataset now has {len(expanded_sequences)} sequences.")

    # Select a percentage of entries from the expanded sequences
    def select_percentage_of_entries(input_list, percentage):
        num_to_select = int(len(input_list) * (percentage / 100))
        selected_items = random.sample(input_list, num_to_select)
        return selected_items

    selected_sequences = select_percentage_of_entries(expanded_sequences, config['dataset']['selection_percentage'])
    print(f"Number of sequences selected for training: {len(selected_sequences)} ({config['dataset']['selection_percentage']}% of the dataset)")

    # Build vocabulary and encode sequences
    tokens = set(" ".join(selected_sequences))
    token_to_idx = {token: idx for idx, token in enumerate(tokens)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}

    def encode_sequence(seq):
        return [token_to_idx[token] for token in seq]

    encoded_sequences = [encode_sequence(seq) for seq in selected_sequences]

    # Split the data into training, validation, and test sets
    total_samples = len(encoded_sequences)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    train_data, val_data, test_data = random_split(
        encoded_sequences, [train_size, val_size, test_size]
    )

    # Dataset class
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

    # Hyperparameters from the config
    seq_length = config['model']['seq_length']
    batch_size = config['model']['batch_size']
    embed_size = config['model']['embed_size']
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']

    filtered_sequences = [seq for seq in encoded_sequences if len(seq) > seq_length]

    # Create datasets
    train_dataset = AminoAcidDataset(train_data, seq_length)
    val_dataset = AminoAcidDataset(val_data, seq_length)
    test_dataset = AminoAcidDataset(test_data, seq_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # LSTM Model
    class LSTMGenerator(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
            super(LSTMGenerator, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden):
            x = self.embedding(x)
            out, hidden = self.lstm(x, hidden)
            out = self.fc(out[:, -1, :])  # Use the output of the last time step
            return out, hidden

        def init_hidden(self, batch_size):
            return (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                    torch.zeros(num_layers, batch_size, hidden_size).to(device))

    # Model, loss, optimizer initialization
    vocab_size = len(token_to_idx)
    model = LSTMGenerator(vocab_size, embed_size, hidden_size, num_layers)
    #model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Function to calculate accuracy
    def calculate_accuracy(outputs, targets):
        _, predicted = torch.max(outputs, dim=1)
        correct = (predicted == targets).float()
        accuracy = correct.sum() / len(correct)
        return accuracy.item()

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        total_train_loss = 0
        total_train_accuracy = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Training") as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                hidden = model.module.init_hidden(inputs.size(0))
                #hidden = model.init_hidden(inputs.size(0))

                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs, targets)

                accuracy = calculate_accuracy(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                total_train_accuracy += accuracy

                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = total_train_accuracy / len(train_loader)
        wandb.log({"train_loss": avg_train_loss, "train_accuracy": avg_train_accuracy})

        model.eval()
        total_val_loss = 0
        total_val_accuracy = 0

        with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Validation") as pbar:
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    hidden = model.module.init_hidden(inputs.size(0))
                    #hidden = model.init_hidden(inputs.size(0))

                    outputs, hidden = model(inputs, hidden)
                    loss = criterion(outputs, targets)

                    accuracy = calculate_accuracy(outputs, targets)

                    total_val_loss += loss.item()
                    total_val_accuracy += accuracy

                    pbar.update(1)

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = total_val_accuracy / len(val_loader)
        wandb.log({"val_loss": avg_val_loss, "val_accuracy": avg_val_accuracy})

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")

    # Testing phase
    model.eval()
    total_test_loss = 0
    total_test_accuracy = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.module.init_hidden(inputs.size(0))
            #hidden = model.init_hidden(inputs.size(0))

            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets)

            accuracy = calculate_accuracy(outputs, targets)

            total_test_loss += loss.item()
            total_test_accuracy += accuracy

    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_accuracy = total_test_accuracy / len(test_loader)
    wandb.log({"test_loss": avg_test_loss, "test_accuracy": avg_test_accuracy})

    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")

if __name__ == "__main__":
    main()