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
    perplexity = math.exp(eval_pred.metrics["eval_loss"]) if "eval_loss" in eval_pred.metrics else float("inf")
    mask = labels != -100
    acc = (np.argmax(logits, axis=-1)[mask] == labels[mask]).mean() if mask.sum() > 0 else 0.0
    return {"accuracy": acc, "perplexity": perplexity}

# ---------------------------------------------------------------------
# 4.  Master training function
# ---------------------------------------------------------------------
def train_lm(config: dict):
    """Trains an MLM, CLM, or custom model on a raw, tokenized dataset."""
    spec = MODEL_ZOO[config["model_key"].lower()]
    objective = spec["objective"]

    if is_main() and config.get("wandb_project"):
        wandb.init(project=config["wandb_project"], config=config)

    # --- Load Data and Tokenizer ---
    print(f"📂 Loading raw dataset from: {config['dataset_path']}")
    ds = load_from_disk(config["dataset_path"])

    tokenizer_path = spec.get("hf_id") or config.get("tokenizer_path")
    if not tokenizer_path:
        raise ValueError("Config must provide 'hf_id' or 'tokenizer_path'.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer `pad_token` set to `eos_token`.")

    # --- Data Preparation ---
    # Prepare labels based on the objective, before removing text columns.
    if objective == "clm":
        print("Transforming dataset for Causal Language Modeling...")
        ds = ds.map(lambda x: {"labels": x["input_ids"]}, num_proc=os.cpu_count())
    elif objective == "custom":
        if "target_ids" in ds["train"].column_names:
            print("Renaming 'target_ids' to 'labels' for custom model training...")
            ds = ds.rename_column("target_ids", "labels")

    # --- CRITICAL STEP: Remove all non-model columns ---
    # This ensures the DataCollator only receives columns it can tensorize.
    model_columns = ["input_ids", "attention_mask", "labels", "token_type_ids"]
    cols_to_remove = [col for col in ds["train"].column_names if col not in model_columns]
    if cols_to_remove:
        print(f"Removing unused columns: {cols_to_remove}")
        ds = ds.remove_columns(cols_to_remove)

    # --- Model and Collator Selection ---
    collator = None
    model = None
    task_type = None
    if objective == "mlm":
        print(f"Loading MLM model: {spec['hf_id']}")
        model = AutoModelForMaskedLM.from_pretrained(spec['hf_id'], trust_remote_code=True)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=config.get("mlm_prob", 0.15))
    elif objective == "clm":
        print(f"Loading CLM model: {spec['hf_id']}")
        model = AutoModelForCausalLM.from_pretrained(spec['hf_id'], trust_remote_code=True)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    else:  # custom
        print(f"Instantiating custom model: {config['model_key']}")
        model_config = config.get("model_config", {})
        if "vocab_size" not in model_config:
            raise ValueError("Custom models require 'vocab_size' in model_config.")
        model = spec["model_class"](**model_config)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if model is None or collator is None:
        raise ValueError(f"Could not initialize model or collator for objective '{objective}'")
    
    if config.get("use_lora", False) and objective == "mlm":
        print("⚡️ Applying LoRA configuration for MLM model...")
        lora_config = LoraConfig(
            # `task_type` is not specified for MLM fine-tuning
            inference_mode=False,
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("lora_target_modules", ["query", "key", "value"])
        )
        model = get_peft_model(model, lora_config)
        print("LoRA Layers Applied:")
        model.print_trainable_parameters()

        for name, param in model.named_parameters():
            if 'embed' in name or 'cls' in name: # 'cls' is the name for the MLM head in BERT-like models
                param.requires_grad = True
    elif config.get("use_lora", False) and objective == "clm":
        # For CLM, the task type is still valid and recommended
        print("⚡️ Applying LoRA configuration for CLM model...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.get("lora_r", 8),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.1),
                # Target modules can be model-specific
                target_modules=config.get("lora_target_modules", ["query", "key", "value"])
            )
        model = get_peft_model(model, lora_config)
        print("LoRA Layers:")
        model.print_trainable_parameters()
    
    model.gradient_checkpointing_enable()

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
        gradient_accumulation_steps=config.get("grad_accum", 1),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False, # Important: we already removed them
    )

    trainer = Trainer( # We can use the standard Trainer again
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
    metrics = trainer.evaluate()
    print("Final evaluation metrics:")
    print(metrics)

    if is_main():
        trainer.save_model(os.path.join(config["checkpoint_path"], "best_model"))
        if wandb.run:
            wandb.log(metrics)
            wandb.finish()

# ---------------------------------------------------------------------
# 5.  CLI Wrapper
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    train_lm(config)