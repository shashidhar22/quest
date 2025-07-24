#! /usr/bin/env python3
import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
import wandb
import evaluate  # type: ignore
from datasets import load_from_disk  # type: ignore
from peft import get_peft_model, LoraConfig  # type: ignore
from transformers import AutoTokenizer, AutoModelForMaskedLM  # type: ignore
from transformers.data.data_collator import DataCollatorForLanguageModeling, DataCollatorWithPadding  # type: ignore
from transformers.training_args import TrainingArguments  # type: ignore
from transformers.trainer import Trainer  # type: ignore
from transformers.trainer_utils import EvalPrediction  # type: ignore
from ray.train.torch import TorchTrainer  # type: ignore
from ray.train import ScalingConfig, RunConfig, FailureConfig, CheckpointConfig  # type: ignore
from ray.air import session  # type: ignore
from typing import Any
from transformers import EarlyStoppingCallback  # type: ignore

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------
# 1. Custom Model Definitions
# ---------------------------------------------------------------------
class CustomRNN(nn.Module):
    """A simple RNN for sequence modeling."""
    def __init__(self, vocab_size: int, embed_size: int = 128, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()  # type: ignore
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **kwargs: dict[str, Any]) -> dict[str, Any]:
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
    def __init__(self, vocab_size: int, embed_size: int = 256, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 512, dropout: float = 0.1, max_len: int = 512):
        super().__init__()  # type: ignore
        self.embedding           = nn.Embedding(vocab_size, embed_size)
        encoder_layer            = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head                = nn.Linear(embed_size, vocab_size)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embed_size))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **kwargs: dict[str, Any]) -> dict[str, Any]:
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}

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
def is_main() -> bool:
    return session.get_world_rank() == 0  # type: ignore

def compute_metrics(eval_pred: Any) -> dict[str, Any]:
    accuracy_metric = evaluate.load("accuracy")  # type: ignore
    logits, labels = eval_pred.predictions, eval_pred.label_ids  # type: ignore
    mask = labels != -100  # type: ignore
    accuracy = accuracy_metric.compute(  # type: ignore
        predictions=np.argmax(logits, axis=-1)[mask],  # type: ignore
        references=labels[mask]  # type: ignore
    )  # type: ignore
    return accuracy  # type: ignore

def train_func(config: dict[str, Any]) -> None:
    spec = MODEL_ZOO[config["model_key"].lower()]
    objective = spec["objective"]

    if is_main() and config.get("wandb_project"):
        wandb.init(project=config["wandb_project"], config=config)

    print(f"Loading pre-tokenized dataset from: {config['dataset_path']}")
    ds = load_from_disk(config["dataset_path"])  # type: ignore
    print("Sorting dataset by length for efficiency...")
    for split in ds:  # type: ignore
        ds[split] = ds[split].map(lambda x: {"len": len(x['input_ids'])}, num_proc=os.cpu_count())  # type: ignore
        ds[split] = ds[split].sort("len")  # type: ignore
        ds[split] = ds[split].remove_columns("len")  # type: ignore

    tokenizer_path = spec.get("hf_id", config.get("model_name"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # type: ignore
    if tokenizer.pad_token is None:  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore

    columns_to_remove = [col for col in ["combo_id", "combo_feats"] if col in ds["train"].column_names]  # type: ignore
    if columns_to_remove:
        print(f"Removing text columns: {columns_to_remove}")
        ds = ds.remove_columns(columns_to_remove)  # type: ignore

    # --- Model and Collator Selection ---
    if objective == "mlm":
        print(f"Loading MLM model: {spec['hf_id']}")
        model = AutoModelForMaskedLM.from_pretrained(spec['hf_id'])  # type: ignore
        print("Using DataCollatorForLanguageModeling for on-the-fly masking.")
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,  # type: ignore
            mlm=True,
            mlm_probability=config.get("mlm_prob", 0.15)
        )  # type: ignore
    elif objective == "custom":
        print(f"Instantiating custom model: {config['model_key']}")
        vocab_size = len(tokenizer)  # type: ignore
        model = spec["model_class"](vocab_size)  # type: ignore
        collator = DataCollatorWithPadding(tokenizer=tokenizer)  # type: ignore
    else:
        raise NotImplementedError(f"Training for objective '{objective}' not fully implemented.")

    if config.get("gradient_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):  # type: ignore
        model.gradient_checkpointing_enable()  # type: ignore

    # Apply LoRA if configured
    lora_applied = False
    if config.get("use_lora", False):
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=config.get("lora_target_modules", ["query", "value"]),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias="none",
        )
        model = get_peft_model(model, lora_config)  # type: ignore
        # LoRA will freeze many weights, but ensure the MLM head is trainable
        for name, param in model.named_parameters():  # type: ignore
            if "cls" in name or "LMPredictionHead" in name:
                param.requires_grad = True
        model.print_trainable_parameters()  # type: ignore
        lora_applied = True
    elif lora_applied is False:
        print("No LoRA configuration provided. Full fine-tuning will be performed.")

    if config.get("test", False):
        print(f"Creating smaller train/val subsets for testing (1000/{len(ds['train'])}, 100/{len(ds['val'])})")  # type: ignore
        train_dataset = ds["train"].shuffle(seed=42).select(range(1000))  # type: ignore
        eval_dataset = ds["val"].shuffle(seed=42).select(range(100))  # type: ignore
    else:
        train_dataset = ds["train"]  # type: ignore
        eval_dataset = ds["val"]  # type: ignore

    training_args = TrainingArguments(
        output_dir=config["checkpoint_path"],
        num_train_epochs=config["num_epochs"],
        label_names=["labels"],
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
        ddp_find_unused_parameters=False,
    )  # type: ignore

    trainer = Trainer(
        model=model,  # type: ignore
        args=training_args,  # type: ignore
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        data_collator=collator,  # type: ignore
        compute_metrics=compute_metrics,  # type: ignore
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # type: ignore
    )  # type: ignore
    print("Starting training...")
    trainer.train()  # type: ignore
    print("Training complete. Saving best model...")
    if is_main():
        best_model_dir = os.path.join(config["checkpoint_path"], "best_model")
        trainer.save_model(best_model_dir)  # type: ignore
        if wandb.run:
            wandb.finish()
        # Sync to S3 if requested
        s3_path = config.get("s3_output_path")
        if s3_path:
            print(f"Syncing best model to S3: {s3_path}")
            sync_cmd = f"aws s3 sync {best_model_dir} {s3_path}"
            ret = os.system(sync_cmd)
            if ret == 0:
                print(f"Best model successfully synced to {s3_path}")
            else:
                print(f"[ERROR] Failed to sync best model to {s3_path}")
    trainer.accelerator.wait_for_everyone()  # type: ignore
    session.report({"status": "done"})  # type: ignore


"""
Ray DDP fine-tuning with fault tolerance.
If interrupted (e.g., spot node recalled), resume with:
    from ray.train.torch import TorchTrainer
    trainer = TorchTrainer.restore('/path/to/ray_results/<your_run_dir>')
    result = trainer.fit()
"""
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--test", action="store_true", help="Run in test mode with a smaller dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of Ray workers (GPUs)")
    parser.add_argument("--s3_output_path", type=str, default=None, help="S3 path to sync best model after training (optional)")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    config["test"] = args.test
    if args.s3_output_path:
        config["s3_output_path"] = args.s3_output_path

    trainer = TorchTrainer(
        train_func,  # type: ignore
        scaling_config=ScalingConfig(
            num_workers=1,  # 1 worker per node/GPU
            use_gpu=True,
            #resources_per_worker={"CPU": 4, "GPU": 1, "memory": 20 * 1024 * 1024 * 1024},
        ),
        run_config=RunConfig(
            failure_config=FailureConfig(max_failures=-1),
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
        train_loop_config=config,
    )  # type: ignore
    result = trainer.fit()  # type: ignore
    print("Ray DDP training finished!", result)

if __name__ == "__main__":
    main() 