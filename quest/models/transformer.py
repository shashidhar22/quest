# transformer.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding as in the original Transformer paper.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)

class DecoderOnlyTransformerGenerator(nn.Module):
    """
    A decoder-only Transformer for causal language modeling:
      - Embedding + PositionalEncoding
      - TransformerEncoder with a causal mask (so it behaves like a GPT-style decoder)
      - Final linear layer to produce vocab logits
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, nhead=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout=dropout)

        # Internally we'll still use a TransformerEncoder, but we supply a causal mask
        # in forward() so tokens cannot attend to future positions.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True  # (batch, seq, embed)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def _generate_causal_mask(self, seq_len, device):
        """
        Return a boolean upper-triangular mask of shape (seq_len, seq_len),
        where True means "block" (no attention to future tokens).
        """
        # mask[i, j] = True if j > i (i.e. upper triangle)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        return mask


    def forward(self, x, attention_mask=None):
        """
        x:           (batch_size, seq_len) of token indices
        attention_mask: (batch_size, seq_len) with 1 = "token present", 0 = "pad"
                        so we can create src_key_padding_mask accordingly.
        Returns: (batch_size, seq_len, vocab_size) of logits
        """
        # 1) Embedding + positional encoding
        emb = self.embedding(x)                 # (batch, seq, embed_size)
        emb = self.pos_encoder(emb)             # add positional encodings

        # 2) Build the (seq_len x seq_len) causal mask
        seq_len = x.size(1)
        causal_mask = self._generate_causal_mask(seq_len, x.device)  # shape (seq_len, seq_len)

        # 3) Build key_padding_mask if attention_mask is provided
        #    The Transformer wants a boolean mask of shape (batch_size, seq_len),
        #    where True = "pad/ignore", False = "keep".
        src_key_padding_mask = None
        if attention_mask is not None:
            # attention_mask=1 means "real token", so invert it to get ignore/keep
            src_key_padding_mask = (attention_mask == 0)

        # 4) Pass through the TransformerEncoder with a causal mask
        #    This effectively makes it operate like a GPT-style decoder.
        #    mask   => [seq_len, seq_len]   (causal)
        #    src_key_padding_mask => [batch_size, seq_len]  (padding)
        transformer_output = self.transformer(
            emb,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch, seq, embed)

        # 5) Final linear projection for next-token prediction
        logits = self.fc_out(transformer_output)  # (batch, seq, vocab_size)
        return logits
