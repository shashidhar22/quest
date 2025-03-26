# config.py

SWEEP_CONFIG = {
    "method": "bayes",  # Bayesian optimization for hyperparameter tuning
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "batch_size": {"values": [64, 128, 256, 512]},
        "data_size": {"values": [0.1, 10, 50, 100]},
        "sequence_length": {"values": [12, 18, 32, 64, 128]},
        "vocab_size": {"values": [27, 50, 100, 1000, 10000]},
        "embedding_dim": {"values": [64, 128, 256, 512]},
        "hidden_dim": {"values": [64, 128, 256, 512]},
        "num_layers": {"values": [1, 2, 3, 4, 5]},
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "num_epochs": {"values": [10, 20, 30, 40]},
        "model_type": {"values": ["lstm"]},
        "resume_training": {"values": [False]},
        "write_checkpoint": {"values": [False]},
    },
}

DEFAULT_CONFIGS = {
    "lstm": {
        "batch_size": 1024,
        "embedding_dim": 256,
        "hidden_dim": 512,
        "num_layers": 3,
        "learning_rate": 1e-4,
        "num_epochs": 80,
        "model_type": "lstm",
        "model_patience": 5,
        "resume_training": False,
        "write_checkpoint": True,
        "early_stop": False,

    },
    "bilstm": {
        "batch_size": 512,
        "embedding_dim": 128,
        "hidden_dim": 512,
        "num_layers": 8,
        "learning_rate": 0.0005,  # BiLSTM might need a lower LR due to bidirectionality
        "num_epochs": 10,
        "model_type": "bilstm"
    },
    "transformer": {
        "model_type": "transformer",
        "embedding_dim": 384,
        "hidden_dim": 1536,
        "num_layers": 12,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "num_epochs": 40,
        "nhead": 12,
        "dropout": 0.2
    },
    "bert": {
        "model_type": "bert",
        "batch_size": 32,       # Often 16, 32, or 64 is typical
        "learning_rate": 5e-5, # A good default LR for BERT fine-tuning
        "num_epochs": 10,
        "resume_training": False,
        "write_checkpoint": True,
        "early_stop": False,
        #"max_len": 256,        # Max length of input text
    }
}
