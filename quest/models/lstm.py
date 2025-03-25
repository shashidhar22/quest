# models/lstm.py

import torch.nn as nn

class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x, attention_mask=None):
        x_emb = self.embedding(x)  # shape: (batch, seq_len, embed_size)
        # Assume you compute lengths from attention_mask, e.g.:
        lengths = attention_mask.sum(dim=1).cpu()  # true lengths
        packed_in = nn.utils.rnn.pack_padded_sequence(x_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_in)
        # Here’s the key change: set total_length so that the output has the same sequence length as x_emb.
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x_emb.size(1))
        logits = self.fc(out)
        return logits
            
        

class BiLSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(BiLSTMGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, attention_mask=None):
        x_emb = self.embedding(x)
        lengths = attention_mask.sum(dim=1).cpu()
        packed_in = nn.utils.rnn.pack_padded_sequence(x_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_in)
        # Set total_length to x_emb.size(1) so that the output has the same sequence length as x
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x_emb.size(1))
        logits = self.fc(out)
        return logits

