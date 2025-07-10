import argparse
import json
import math
import os
import pathlib
import torch
import torch.nn as nn
import numpy as np
import wandb
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType
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

# This is important to prevent hangs in multiprocessing with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------
# 1. Custom Model Definitions
# ---------------------------------------------------------------------
# --- Create a new, robust data collator class ---
class RobustMLMCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        # First, let the original collator do its job of creating the masked `labels`.
        # We set `return_tensors="np"` to get NumPy arrays without padding.
        batch = super().__call__(examples)

        # Now, we manually pad the batch to the max length in this batch.
        # This bypasses the buggy internal padding logic.
        max_length = max(len(x) for x in batch["input_ids"])

        padded_batch = {}
        for key, value in batch.items():
            padded_sequences = []
            # Use -100 for labels padding, and the tokenizer's pad_id for others
            padding_value = -100 if key == "labels" else self.tokenizer.pad_token_id
            
            for sequence in value:
                padding_needed = max_length - len(sequence)
                padded_sequences.append(sequence.tolist() + [padding_value] * padding_needed)
            
            padded_batch[key] = torch.tensor(padded_sequences)
            
        return padded_batch
    
class FastMLMCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        # 1. Let the base collator handle the masking.
        # It returns un-padded numpy arrays because of `return_tensors="np"`.
        batch = super().__call__(examples)

        # 2. Find the longest sequence in this batch
        max_length = max(len(x) for x in batch["input_ids"])

        # 3. Use the tokenizer's highly optimized .pad() method.
        # This performs padding in fast, compiled code, not slow Python loops.
        batch = self.tokenizer.pad(
            batch,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt", # Return PyTorch tensors
        )
        return batch

class CustomRNN(nn.Module):
    """A simple RNN for sequence modeling."""
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        x, _ = self.rnn(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}

class CustomTransformer(nn.Module):
    """A simple Transformer Encoder for sequence modeling."""
    def __init__(self, vocab_size, embed_size=256, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_size, vocab_size)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embed_size))

    def forward(self, input_ids, labels=None, **kwargs):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}

class CompileFriendlyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        This method is overridden to remove the `num_items_in_batch` argument
        that is incompatible with a torch.compiled model.
        """
        # The Trainer internally adds this for its own bookkeeping
        if "num_items_in_batch" in inputs:
            inputs.pop("num_items_in_batch")
            
        return super().compute_loss(model, inputs, return_outputs)
# ---------------------------------------------------------------------
# 2.  Model Zoo
# ---------------------------------------------------------------------
MODEL_ZOO = {
    "bert":     {"hf_id": "bert-base-cased", "objective": "mlm"},
    "gpt2":     {"hf_id": "gpt2", "objective": "clm"},
    "llama3":   {"hf_id": "meta-llama/Meta-Llama-3-8B", "objective": "clm"},
    "esm":      {"hf_id": "facebook/esm2_t33_650M_UR50D", "objective": "mlm"},
    "protbert": {"hf_id": "Rostlab/prot_bert_bfd", "objective": "mlm"},
    "custom-rnn": {"model_class": CustomRNN, "objective": "custom"},
    "custom-transformer": {"model_class": CustomTransformer, "objective": "custom"},
}

# ---------------------------------------------------------------------
# 3.  Callbacks and Metric Helpers
# ---------------------------------------------------------------------
class ClearCudaCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

def is_main():
    return int(os.environ.get("RANK", 0)) == 0

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # The trainer already calculates loss, so we can access it from metrics
    loss = eval_pred.metrics.get("eval_loss", 0.0)
    perplexity = math.exp(loss) if loss < 20 else float("inf")
    mask = labels != -100
    acc = (np.argmax(logits, axis=-1)[mask] == labels[mask]).mean() if mask.sum() > 0 else 0.0
    return {"accuracy": acc, "perplexity": perplexity}

# ---------------------------------------------------------------------
# Master training function
# ---------------------------------------------------------------------
def train_lm(config: dict):
    """Trains a model on a pre-processed dataset."""
    spec = MODEL_ZOO[config["model_key"].lower()]
    objective = spec["objective"]

    if is_main() and config.get("wandb_project"):
        wandb.init(project=config["wandb_project"], config=config)

    # --- Load Processed Data and Tokenizer ---
    print(f"📂 Loading PROCESSED dataset from: {config['dataset_path']}")
    ds = load_from_disk(config["dataset_path"])
    tokenizer_path = spec.get("hf_id")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Remove unused text columns immediately after loading ---
    # This ensures only tokenized data is passed to the Trainer.
    columns_to_remove = [col for col in ["combo_id", "combo_feats"] if col in ds["train"].column_names]
    if columns_to_remove:
        print(f"Removing text columns: {columns_to_remove}")
        ds = ds.remove_columns(columns_to_remove)

    # --- Model and Collator Selection ---
    if objective == "mlm":
        print(f"Loading MLM model: {spec['hf_id']}")
        model = AutoModelForMaskedLM.from_pretrained(spec['hf_id'])
        
        # --- MODIFICATION: Use our new robust collator ---
        print("Using DataCollatorForLanguageModeling for on-the-fly masking.")
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=True, 
            mlm_probability=config.get("mlm_prob", 0.15)
        )
    else:
        # Handle CLM or other objectives if needed
        raise NotImplementedError(f"Training for objective '{objective}' not fully implemented.")


    model.gradient_checkpointing_enable()

    # Apply LoRA if configured
    if config.get("use_lora", False):
        print("⚡️ Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=config.get("lora_target_modules", ["query", "value"]),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        
        # Make the MLM head trainable
        for name, param in model.named_parameters():
            if 'cls' in name or 'LMPredictionHead' in name:
                param.requires_grad = True
        model.print_trainable_parameters()

    # --- Trainer Setup ---
    training_args = TrainingArguments(
        output_dir=config["checkpoint_path"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config.get("eval_batch_size", config["batch_size"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        report_to="wandb" if config.get("wandb_project") else "none",
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", True),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[ClearCudaCacheCallback()],
    )

    # --- Train and Evaluate ---
    print("🚀 Starting training...")
    trainer.train()
    
    print("🏁 Training complete. Evaluating final model...")
    metrics = trainer.evaluate(eval_dataset=ds["test"]) # Evaluate on the test set
    print("Final test metrics:")
    print(metrics)

    if is_main():
        trainer.save_model(os.path.join(config["checkpoint_path"], "best_model"))
        if wandb.run:
            wandb.log({"test_metrics": metrics})
            wandb.finish()


# --- CLI Wrapper ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    train_lm(config)