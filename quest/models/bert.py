import os
import wandb
from datasets import load_from_disk
import math
import numpy as np
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from transformers import BertConfig

from transformers import TrainerCallback

class ClearCudaCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        import torch
        torch.cuda.empty_cache()
        return control

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    # Get predictions from logits
    predictions = np.argmax(logits, axis=-1)
    # For masked language modeling, labels that are ignored are set to -100.
    # Create a mask to filter out those tokens.
    mask = labels != -100
    if mask.sum() > 0:
        accuracy = (predictions[mask] == labels[mask]).mean()
    else:
        accuracy = 0.0
    # Perplexity is simply the exponential of the loss.
    # Here, we assume that the evaluation loss is low enough (e.g. <20) to avoid overflow.
    perplexity = math.exp(eval_pred.metrics["eval_loss"]) if "eval_loss" in eval_pred.metrics and eval_pred.metrics["eval_loss"] < 20 else float('inf')
    return {"accuracy": accuracy, "perplexity": perplexity}

def train_bert_with_transformers(config):
    """
    Train a BERT Masked LM with Hugging Face Transformers, logging metrics to W&B.
    """

    # 1) Initialize W&B (if not already done)
    #    If your code is already calling wandb.init(...) in your Trainer class or elsewhere,
    #    you can optionally skip or guard this to avoid multiple inits.
    if is_main_process() and wandb.run is None:  # e.g., if W&B isn't initialized yet
        wandb.init(project="quest", config=config)

    # 2) Load the dataset you prepared for BERT
    #    Typically, these are raw-text columns or pre-tokenized columns.
    dataset = load_from_disk(config["dataset"])  # "train", "validation", "test"

    # 3) Load / create a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # 4) If your dataset is raw text, map it:
    def tokenize_function(examples):
        return tokenizer(
            examples["sequence"], 
            truncation=True, 
            padding="max_length", 
            max_length=512
        )

    # Only do this if the dataset still has raw text in a column named "sequence":
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # 5) Define a BERT MaskedLM model (or BertForPreTraining, if you prefer)
    #initialization = BertConfig.from_pretrained("bert-base-cased")
    #model = BertForMaskedLM(initialization)
    model = BertForMaskedLM.from_pretrained("bert-base-cased")

    # 6) Data Collator to handle dynamic masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 7) Configure Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir=config["checkpoint_path"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        eval_strategy="epoch",    # Evaluate every epoch
        save_strategy="epoch",          # Save every epoch
        logging_steps=100,              # How often to log
        report_to="wandb",              # KEY: sends HF Trainer logs to W&B
        run_name=wandb.run.name if wandb.run else None,
        fp16=True,
        # Add more HF Trainer settings as needed:
        #   fp16=True,  # If you want Automatic Mixed Precision
        #   gradient_accumulation_steps=...
        #   etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[ClearCudaCacheCallback()]
    )

    # 8) Train
    trainer.train()

    # 9) Evaluate
    eval_metrics = trainer.evaluate()
    print("Eval metrics:", eval_metrics)

    # 10) Optionally, log more custom items to W&B
    wandb.log({"eval_loss": eval_metrics["eval_loss"]})

    # 11) Finish the W&B run if you started it in this function
    wandb.finish()
