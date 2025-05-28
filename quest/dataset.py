# dataset.py

import torch
from torch.utils.data import Dataset
from quest.constants import PAD_TOKEN_ID, PAD_TOKEN_STR

class AminoAcidDataset(Dataset):
    """
    A flexible dataset that:
    - Tokenizes raw sequences with a BPE tokenizer
    - Depending on `model_type`, generates:
      (A) RNN-style sliding windows (LSTM/BiLSTM)
      (B) Transformer-style causal language modeling with entire sequence

    For the Transformer mode, the dataset uses the *entire* sequence as a single chunk
    (no segmentation). Optionally truncate if it's longer than `seq_length`.
    """

    def __init__(
        self,
        sequences,
        tokenizer,
        seq_length=128,
        model_type="rnn",
        step=1,
        truncate_long_sequences=True
    ):
        """
        Args:
            sequences (List[str]):
                A list of raw amino-acid strings (with start/stop tokens, etc.)
            tokenizer (tokenizers.Tokenizer):
                A pre-trained Hugging Face BPE tokenizer
            seq_length (int):
                The fixed length for each sample (input/target).
            model_type (str):
                - 'rnn' for LSTM/BiLSTM sliding-window style
                - 'transformer' for decoder-only causal LM (no chunking)
            step (int):
                Stride used for sliding window in 'rnn' mode
            truncate_long_sequences (bool):
                If True (Transformer mode), sequences longer than seq_length are truncated;
                if False, you'll keep the entire sequence, ignoring seq_length. 
                (Be careful with large GPU memory usage for very long sequences.)
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.model_type = model_type.lower()
        self.step = step
        self.truncate_long_sequences = truncate_long_sequences

        # Attempt to find the pad token ID in the tokenizer vocab
        self.pad_token_id = tokenizer.token_to_id(PAD_TOKEN_STR)
        if self.pad_token_id is None:
            self.pad_token_id = PAD_TOKEN_ID  # fallback

        self.samples = []
        self._build_samples(sequences)

    def __repr__(self):
        """
        Human-readable summary that shows the dataset size and (for a quick sanity
        check) the first sample’s decoded input and target strings.
        """
        header = f"{self.__class__.__name__}(num_samples={len(self)})"
        if not self.samples:                     # empty dataset
            return header

        # Grab the first (X, Y) pair and decode token-ids → string
        x_ids, y_ids = self.samples[0]
        input_str  = self.tokenizer.decode(x_ids, skip_special_tokens=True)
        target_str = self.tokenizer.decode(y_ids, skip_special_tokens=True)

        return (
            f"{header}\n"
            f"  input : {input_str}\n"
            f"  target: {target_str}"
        )


    def _build_samples(self, sequences):
        if self.model_type == "rnn":
            self._build_rnn_samples(sequences)
        elif self.model_type == "transformer":
            self._build_transformer_samples_entire(sequences)
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. Must be 'rnn' or 'transformer'."
            )

    def _build_rnn_samples(self, sequences):
        """
        RNN (LSTM/BiLSTM) mode:
        Use a sliding-window approach over the tokenized sequence:
          - If len(tokens) >= seq_length+1:
              create multiple windows, each size (seq_length+1).
              X = window[:-1], Y = window[1:]
          - If too short, pad up to seq_length+1 and do single window.
        """
        for seq_str in sequences:
            if not seq_str or not isinstance(seq_str, str):
                continue

            encoding = self.tokenizer.encode(seq_str)
            tokens = encoding.ids

            # If tokens are long enough for a sliding window
            if len(tokens) >= self.seq_length + 1:
                num_subseqs = (len(tokens) - (self.seq_length + 1)) // self.step + 1
                for i in range(num_subseqs):
                    start = i * self.step
                    end = start + self.seq_length + 1
                    subseq = tokens[start:end]
                    X = subseq[:-1]
                    Y = subseq[1:]
                    self.samples.append((X, Y))
            else:
                # Too short => single padded sample
                needed = (self.seq_length + 1) - len(tokens)
                padded_seq = tokens + [self.pad_token_id] * needed
                X = padded_seq[:-1]
                Y = padded_seq[1:]
                self.samples.append((X, Y))

    def _build_transformer_samples_entire(self, sequences):
        """
        Transformer (decoder-only) mode (No chunking):
        - Each entire tokenized sequence -> single example.
        - If shorter than seq_length, pad up to seq_length.
        - If longer and `truncate_long_sequences=True`, truncate to seq_length.
          Otherwise, keep entire length (which can exceed seq_length).
        - Then X = chunk[:-1], Y = chunk[1:] for causal LM shift.
        """
        for seq_str in sequences:
            if not seq_str or not isinstance(seq_str, str):
                continue

            encoding = self.tokenizer.encode(seq_str)
            tokens = encoding.ids

            # Handle length > seq_length
            if len(tokens) > self.seq_length:
                if self.truncate_long_sequences:
                    tokens = tokens[:self.seq_length]
                # else keep entire length (warning: can be huge)

            # Pad if shorter
            if len(tokens) < self.seq_length:
                needed = self.seq_length - len(tokens)
                tokens = tokens + [self.pad_token_id] * needed

            # If there's only 1 token after possible truncation/padding, skip
            if len(tokens) < 2:
                continue

            # Causal LM shift
            X = tokens[:-1]  # all but last
            Y = tokens[1:]   # all but first
            self.samples.append((X, Y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x_tensor = torch.tensor(x, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor
