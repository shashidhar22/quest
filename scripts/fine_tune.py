import os, math, numpy as np, wandb, torch, time
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerCallback,
)
import argparse, json, pathlib
import torch.nn as nn



# ---------------------------------------------------------------------
# 1. Custom Model Definitions (for custom-rnn & custom-transformer)
# ---------------------------------------------------------------------

class CustomRNN(nn.Module):
    """A simple RNN for sequence modeling."""
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x, _ = self.rnn(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            # Flatten the logits and labels for CrossEntropyLoss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}

class CustomTransformer(nn.Module):
    """A simple Transformer Encoder for sequence modeling."""
    def __init__(self, vocab_size, embed_size=256, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_size, vocab_size)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, embed_size)) # Simple learned positional encoding

    def forward(self, input_ids, labels=None):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_encoder[:, :seq_len, :] # Add positional encoding
        x = self.transformer_encoder(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------
# 2.  Updated Model Zoo
# ---------------------------------------------------------------------
MODEL_ZOO = {
    # --- Hugging Face Models ---
    "bert":     {"hf_id": "bert-base-cased", "objective": "mlm"},
    "gpt2":     {"hf_id": "gpt2", "objective": "clm"},
    "llama3":   {"hf_id": "meta-llama/Meta-Llama-3-8B", "objective": "clm"}, # Note: 8B not 7B
    "esm":      {"hf_id": "facebook/esm2_t33_650M_UR50D", "objective": "mlm"}, # Example ESM-2
    "protbert": {"hf_id": "Rostlab/prot_bert_bfd", "objective": "mlm"},

    # --- Custom Models ---
    "custom-rnn": {"model_class": CustomRNN, "objective": "custom"},
    "custom-transformer": {"model_class": CustomTransformer, "objective": "custom"},
}

# ---------------------------------------------------------------------
# 3.  Callbacks and Metric Helpers
# ---------------------------------------------------------------------
class ClearCudaCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        return control

def is_main():
    return int(os.environ.get("RANK", 0)) == 0

def compute_metrics(eval_pred: EvalPrediction):
    # This function now needs to handle the output of our custom models
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Perplexity calculation remains the same
    perplexity = math.exp(eval_pred.metrics["eval_loss"]) if "eval_loss" in eval_pred.metrics and eval_pred.metrics["eval_loss"] < 20 else float("inf")

    # Accuracy calculation (works for both MLM and our custom models)
    mask = labels != -100
    if mask.sum() > 0:
        acc = (np.argmax(logits, axis=-1)[mask] == labels[mask]).mean()
    else:
        acc = 0.0
    return {"accuracy": acc, "perplexity": perplexity}


# ---------------------------------------------------------------------
# 4.  Master training function
# ---------------------------------------------------------------------
def train_lm(config: dict):
    """Train an MLM, CLM, or custom model picked via `config["model_key"]`."""

    spec = MODEL_ZOO[config["model_key"].lower()]
    objective = spec["objective"]

    if is_main() and wandb.run is None:
        wandb.init(project="quest", config=config)

    print("Loading pre-tokenized dataset from disk...")
    ds = load_from_disk(config["dataset"])

    # We still need a tokenizer for the DataCollator, even if not re-tokenizing
    # For custom models, we load the BPE tokenizer you trained.
    # For HF models, we load the standard one.
    tokenizer_path = spec.get("hf_id") or config.get("tokenizer_path")
    if not tokenizer_path:
        raise ValueError("Config must provide 'hf_id' for HF models or 'tokenizer_path' for custom models.")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Tokenizer missing pad token; setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # --- Prepare columns and labels based on objective ---
    if objective == "custom":
        # For our custom models, the datawriter created 'input_ids' and 'target_ids'.
        # The Trainer expects the label column to be named 'labels'.
        if "target_ids" in ds["train"].column_names:
            ds = ds.rename_column("target_ids", "labels")
        ds.set_format("torch", columns=["input_ids", "labels"])
    elif objective == "clm":
        # For CLM, labels are the input_ids. The collator will shift them.
        ds = ds.map(lambda x: {"labels": x["input_ids"]})
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    else: # mlm
        # For MLM, the data collator creates the labels by masking input_ids on the fly.
        ds.set_format("torch", columns=["input_ids", "attention_mask"])

    # ----------  Model and Collator  ----------
    if objective == "mlm":
        print(f"Loading MLM model: {spec['hf_id']}")
        model = AutoModelForMaskedLM.from_pretrained(spec['hf_id'], trust_remote_code=True)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    elif objective == "clm":
        print(f"Loading CLM model: {spec['hf_id']}")
        model = AutoModelForCausalLM.from_pretrained(spec['hf_id'], trust_remote_code=True)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    else: # custom
        print(f"Instantiating custom model: {config['model_key']}")
        # Custom models need their config passed from the training config file
        model_config = config.get("model_config", {})
        if "vocab_size" not in model_config:
             raise ValueError("Custom models require 'vocab_size' in model_config.")
        model = spec["model_class"](**model_config)
        # For our custom models, we just need to pad batches to the same length.
        collator = DataCollatorWithPadding(tokenizer)
    
    # Enable gradient checkpointing for all models to save memory
    model.gradient_checkpointing_enable()

    # ----------  Trainer args  ----------
    targs = TrainingArguments(
        output_dir=config["checkpoint_path"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        report_to="wandb",
        fp16=True, # Ensure your GPU supports this
        gradient_accumulation_steps=config.get("grad_accum", 1),
        run_name=wandb.run.name if wandb.run else None,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["val"], # --- Use 'val' split from datawriter ---
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[ClearCudaCacheCallback()],
    )

    # ----------  Train / evaluate  ----------
    trainer.train()
    metrics = trainer.evaluate()
    print("Final evaluation metrics:")
    print(metrics)

    if is_main():
        trainer.save_model(os.path.join(config["checkpoint_path"], "best_model"))
        wandb.log(metrics)
        wandb.finish()


# ---------------------------------------------------------------------
# 5.  Example CLI wrapper (Unchanged)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    cfg = json.loads(cfg_path.read_text())

    train_lm(cfg)