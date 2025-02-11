# models/factory.py

from quest.models.lstm import LSTMGenerator, BiLSTMGenerator

def get_model(model_type, vocab_size, embed_size, hidden_size, num_layers):
    if model_type == "lstm":
        return LSTMGenerator(vocab_size, embed_size, hidden_size, num_layers)
    elif model_type == "bilstm":
        return BiLSTMGenerator(vocab_size, embed_size, hidden_size, num_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
