# config.py

SWEEP_CONFIG = {
    "method": "bayes",  # Bayesian optimization for hyperparameter tuning
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "batch_size": {"values": [64, 128, 256]},
        "embedding_dim": {"values": [64, 128, 256]},
        "hidden_dim": {"values": [64, 128, 256]},
        "num_layers": {"values": [1, 2, 3]},
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "num_epochs": {"values": [10, 20, 30, 40]},
        "model_type": {"values": ["lstm", "bilstm"]},
    },
}

DEFAULT_CONFIGS = {
    "lstm": {
        "batch_size": 256,
        "embedding_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "learning_rate": 0.001,
        "num_epochs": 10,
        "model_type": "lstm"
    },
    "bilstm": {
        "batch_size": 1024,
        "embedding_dim": 128,
        "hidden_dim": 1024,
        "num_layers": 8,
        "learning_rate": 0.0005,  # BiLSTM might need a lower LR due to bidirectionality
        "num_epochs": 10,
        "model_type": "bilstm"
    }
}
