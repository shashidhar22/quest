import argparse
import json
import math
import os
import torch
import numpy as np
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerCallback,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Model Zoo to map config key to Hugging Face ID ---
MODEL_ZOO = {
    "bert":     {"hf_id": "bert-base-cased"},
    "protbert": {"hf_id": "Rostlab/prot_bert_bfd"},
    "esm":      {"hf_id": "facebook/esm2_t33_650M_UR50D"},
    # Add other models as needed
}

# --- Callbacks and Metric Helpers ---
class ClearCudaCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    loss = eval_pred.metrics.get("eval_loss", 0.0)
    perplexity = math.exp(loss) if loss > 0 else float("inf")
    mask = labels != -100
    acc = (np.argmax(logits, axis=-1)[mask] == labels[mask]).mean() if mask.sum() > 0 else 0.0
    return {"accuracy": acc, "perplexity": perplexity}

# --- Main Training Function ---
def train_lm(config: dict):
    # --- MODIFICATION: Look up model details from MODEL_ZOO ---
    spec = MODEL_ZOO[config["model_key"].lower()]
    model_hf_id = spec['hf_id']

    # --- Load Model, Tokenizer, and PROCESSED Data ---
    print(f"📂 Loading PROCESSED dataset from: {config['dataset_path']}")
    ds = load_from_disk(config['dataset_path'])
    
    print(f"🔄 Loading model and tokenizer for: {model_hf_id}")
    model = AutoModelForMaskedLM.from_pretrained(model_hf_id)
    tokenizer = AutoTokenizer.from_pretrained(model_hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The data is pre-masked, so we only need to pad batches.
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- LoRA Configuration ---
    if config.get("use_lora", False):
        print("⚡️ Applying LoRA configuration...")
        peft_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=config.get("lora_target_modules", ["query", "value"]),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        
        # Make the MLM head trainable
        for name, param in model.named_parameters():
            if 'cls' in name or 'LMPredictionHead' in name:
                param.requires_grad = True
        model.print_trainable_parameters()
    
    # --- Trainer Setup ---
    training_args = TrainingArguments(
        output_dir=config["checkpoint_path"],
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 16),
        per_device_eval_batch_size=config.get("eval_batch_size", 32),
        gradient_accumulation_steps=config.get("grad_accum", 1),
        learning_rate=config.get("learning_rate", 2e-5),
        weight_decay=config.get("weight_decay", 0.01),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        fp16=config.get("fp16", True),
        bf16=config.get("bf16", False),
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

    print("🏁 Training complete. Evaluating on test set...")
    metrics = trainer.evaluate(eval_dataset=ds["test"])
    print("Final test metrics:", metrics)

    if int(os.environ.get("RANK", 0)) == 0:
        trainer.save_model(os.path.join(config["checkpoint_path"], "best_model"))

# --- CLI Wrapper ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model on a PRE-PROCESSED dataset.")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    train_lm(config)