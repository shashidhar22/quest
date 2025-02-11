# dataset.py

import torch
from torch.utils.data import Dataset

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
        raise IndexError("Index out of range.")
