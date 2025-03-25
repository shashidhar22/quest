# models/factory.py

from quest.models.lstm import LSTMGenerator, BiLSTMGenerator
from quest.models.transformer import DecoderOnlyTransformerGenerator
def get_model(model_type, vocab_size, embed_size, hidden_size, num_layers, nhead=0, dropout=0):
    if model_type == "lstm":
        return LSTMGenerator(vocab_size, embed_size, hidden_size, num_layers)
    elif model_type == "bilstm":
        return BiLSTMGenerator(vocab_size, embed_size, hidden_size, num_layers)
    elif model_type == "transformer":
        # Now returns the new decoder-only model
        return DecoderOnlyTransformerGenerator(
            vocab_size, embed_size, hidden_size,
            num_layers, nhead, dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
