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
from peft import get_peft_model, LoraConfig, PeftModel  # type: ignore
from transformers import AutoTokenizer, AutoModelForMaskedLM  # type: ignore
from transformers.data.data_collator import DataCollatorForLanguageModeling, DataCollatorWithPadding  # type: ignore
import re
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
# 3. Custom Data Collator for Mode-Specific Masking
# ---------------------------------------------------------------------
class CustomMaskingDataCollator:
    """Custom data collator that applies different masking strategies based on mode."""

    def __init__(self, tokenizer, mode: str = "default", mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mode = mode.lower()
        self.mlm_probability = mlm_probability

    def __call__(self, features):
        batch = {}

        # Get the first feature to determine structure
        first_feature = features[0]

        # Handle different input formats
        if "input_ids" in first_feature:
            input_ids = [f["input_ids"] for f in features]
        else:
            raise ValueError("Features must contain 'input_ids'")

        # Get combo_feats if available
        combo_feats_list = None
        if "combo_feats" in first_feature:
            combo_feats_list = [f["combo_feats"] for f in features]

        # Apply custom masking based on mode
        if self.mode in ["tra", "trb"]:
            masked_inputs, labels = self._mask_tra_trb_mode(input_ids)
        elif self.mode != "default" and combo_feats_list:
            masked_inputs, labels = self._mask_multi_combo_mode(input_ids, combo_feats_list)
        else:
            # Default MLM masking
            masked_inputs, labels = self._default_mlm_masking(input_ids)

        batch["input_ids"] = torch.tensor(masked_inputs, dtype=torch.long)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        # Add attention mask
        batch["attention_mask"] = torch.tensor([
            [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in seq]
            for seq in masked_inputs
        ], dtype=torch.long)

        return batch

    def _mask_tra_trb_mode(self, input_ids_list):
        """Mask everything except first 4 and last 4 amino acids for TRA/TRB modes."""
        masked_inputs = []
        labels = []

        for input_ids in input_ids_list:
            input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            label_ids = [-100] * len(input_ids)

            # Find actual sequence boundaries (exclude special tokens)
            seq_start = 0
            seq_end = len(input_ids)

            # Skip special tokens at the beginning
            for i, token_id in enumerate(input_ids):
                token = self.tokenizer.decode([token_id])
                if not token.startswith('[') and token not in ['<s>', '</s>', '<pad>']:
                    seq_start = i
                    break

            # Find end of sequence (before padding/special tokens)
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] != self.tokenizer.pad_token_id:
                    token = self.tokenizer.decode([input_ids[i]])
                    if not token.startswith('[') and token not in ['<s>', '</s>', '<pad>']:
                        seq_end = i + 1
                        break

            # Mask middle section, keeping first 4 and last 4 amino acids
            preserve_start = min(seq_start + 4, seq_end)
            preserve_end = max(seq_end - 4, preserve_start)

            if preserve_end > preserve_start:
                # Mask the middle section
                for i in range(preserve_start, preserve_end):
                    if input_ids[i] != self.tokenizer.pad_token_id:
                        label_ids[i] = input_ids[i]  # Store original for loss calculation
                        input_ids[i] = self.tokenizer.mask_token_id

            masked_inputs.append(input_ids)
            labels.append(label_ids)

        return masked_inputs, labels

    def _mask_multi_combo_mode(self, input_ids_list, combo_feats_list):
        """Mask the second complete sequence when multiple combo_feats are present."""
        masked_inputs = []
        labels = []

        for input_ids, combo_feats in zip(input_ids_list, combo_feats_list):
            input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            label_ids = [-100] * len(input_ids)

            # Only apply special masking if there are multiple features
            if len(combo_feats) > 1:
                # Find separator tokens to identify sequence boundaries
                separators = []
                for i, token_id in enumerate(input_ids):
                    token = self.tokenizer.decode([token_id])
                    if '[SEP]' in token or token in ['[ETRA]', '[ETRB]', '[EPEP]', '[EMHO]', '[EMHT]']:
                        separators.append(i)

                if len(separators) >= 1:
                    # Mask from first separator to second separator (or end if only one separator)
                    mask_start = separators[0] + 1
                    mask_end = separators[1] if len(separators) > 1 else len(input_ids)

                    # Find actual end of sequence (before padding)
                    for i in range(len(input_ids) - 1, -1, -1):
                        if input_ids[i] != self.tokenizer.pad_token_id:
                            mask_end = min(mask_end, i + 1)
                            break

                    # Apply masking to the second sequence
                    for i in range(mask_start, mask_end):
                        if input_ids[i] != self.tokenizer.pad_token_id:
                            label_ids[i] = input_ids[i]
                            input_ids[i] = self.tokenizer.mask_token_id
                else:
                    # Fallback: mask random 15% if no separators found
                    input_ids, label_ids = self._apply_random_masking(input_ids, label_ids)
            else:
                # Single feature: apply random masking
                input_ids, label_ids = self._apply_random_masking(input_ids, label_ids)

            masked_inputs.append(input_ids)
            labels.append(label_ids)

        return masked_inputs, labels

    def _default_mlm_masking(self, input_ids_list):
        """Apply standard 15% random masking."""
        masked_inputs = []
        labels = []

        for input_ids in input_ids_list:
            input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            label_ids = [-100] * len(input_ids)

            input_ids, label_ids = self._apply_random_masking(input_ids, label_ids)

            masked_inputs.append(input_ids)
            labels.append(label_ids)

        return masked_inputs, labels

    def _apply_random_masking(self, input_ids, label_ids):
        """Apply random 15% masking to non-special tokens."""
        import random

        for i, token_id in enumerate(input_ids):
            if token_id == self.tokenizer.pad_token_id:
                continue

            # Skip special tokens
            token = self.tokenizer.decode([token_id])
            if token.startswith('[') or token in ['<s>', '</s>', '<pad>']:
                continue

            if random.random() < self.mlm_probability:
                label_ids[i] = input_ids[i]

                # 80% mask, 10% random, 10% keep
                prob = random.random()
                if prob < 0.8:
                    input_ids[i] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    input_ids[i] = random.randint(0, len(self.tokenizer) - 1)
                # else keep original

        return input_ids, label_ids

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
    # Check if using local model path or MODEL_ZOO
    model_path = config.get("model_path")
    if model_path:
        # Using local model - determine objective from config or default to MLM
        objective = config.get("objective", "mlm")
        spec = {"objective": objective}
        print(f"Using local model path: {model_path} with objective: {objective}")
    else:
        # Using MODEL_ZOO
        model_key = config["model_key"].lower()
        if model_key not in MODEL_ZOO:
            raise ValueError(f"Unknown model_key: {model_key}. Available models: {list(MODEL_ZOO.keys())}")
        spec = MODEL_ZOO[model_key]
        objective = spec["objective"]
        print(f"Using MODEL_ZOO model: {model_key} with objective: {objective}")

    # Get masking mode from config (default to "default")
    masking_mode = config.get("masking_mode", "default")

    if is_main() and config.get("wandb_project"):
        wandb.init(project=config["wandb_project"], config=config)

    dataset_path = config["dataset_path"]
    print(f"Loading pre-tokenized dataset from: {dataset_path}")
    
    # Handle S3 paths
    if dataset_path.startswith("s3://"):
        print("Loading dataset from S3...")
        import s3fs
        # s3fs is already installed via setup_commands
        ds = load_from_disk(dataset_path)  # type: ignore
    else:
        print("Loading dataset from local path...")
        ds = load_from_disk(dataset_path)  # type: ignore
    
    # Sample the dataset if sample_rate is provided
    sample_rate = config.get("sample_rate")
    if sample_rate is not None and 0 < sample_rate < 1:
        print(f"Sampling dataset with rate: {sample_rate}")
        for split in ds:  # type: ignore
            original_size = len(ds[split])  # type: ignore
            sample_size = int(original_size * sample_rate)
            print(f"Sampling {split}: {original_size} -> {sample_size} examples")
            ds[split] = ds[split].shuffle(seed=42).select(range(sample_size))  # type: ignore
    elif sample_rate is not None:
        print(f"Invalid sample_rate: {sample_rate}. Must be between 0 and 1. Using full dataset.")
    
    print("Sorting dataset by length for efficiency...")
    for split in ds:  # type: ignore
        ds[split] = ds[split].map(lambda x: {"len": len(x['input_ids'])}, num_proc=os.cpu_count())  # type: ignore
        ds[split] = ds[split].sort("len")  # type: ignore
        ds[split] = ds[split].remove_columns("len")  # type: ignore

    # Set tokenizer path based on whether using local model or MODEL_ZOO
    if model_path:
        tokenizer_path = model_path
        print(f"Loading tokenizer from local path: {model_path}")
    else:
        tokenizer_path = spec.get("hf_id", config.get("model_name"))
        print(f"Loading tokenizer from Hugging Face: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # type: ignore
    if tokenizer.pad_token is None:  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore

    # Print current dataset structure for debugging
    print(f"Dataset columns: {ds['train'].column_names}")  # type: ignore
    print(f"Sample data types: {ds['train'].features}")  # type: ignore

    # Remove any non-tensor columns that shouldn't be passed to the model
    columns_to_remove = [col for col in ["combo_id", "combo_feats"] if col in ds["train"].column_names]  # type: ignore
    if columns_to_remove:
        print(f"Removing text columns: {columns_to_remove}")
        ds = ds.remove_columns(columns_to_remove)  # type: ignore
        print(f"Dataset columns after removal: {ds['train'].column_names}")  # type: ignore

    # --- Model and Collator Selection ---
    if objective == "mlm":
        # Check if model_path is a PEFT adapter
        is_peft_input = False
        if model_path and os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"Detected PEFT adapter at {model_path}")
            is_peft_input = True

            # Determine base model for the PEFT adapter
            base_model_name = config.get("base_model")
            if base_model_name:
                print(f"Using user-specified base model: {base_model_name}")
            else:
                # Try to get from adapter config
                try:
                    import json
                    with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                        adapter_config = json.load(f)
                    base_model_name = adapter_config.get("base_model_name_or_path")
                    if base_model_name:
                        print(f"Using base model from adapter config: {base_model_name}")
                    else:
                        base_model_name = "Rostlab/prot_bert"
                        print("No base model in config, defaulting to Rostlab/prot_bert")
                except Exception as e:
                    print(f"Could not read adapter config: {e}")
                    base_model_name = "Rostlab/prot_bert"
                    print("Defaulting to Rostlab/prot_bert")

            # Load base model and apply existing PEFT adapter, then merge weights
            print(f"Loading base model: {base_model_name}")
            base_model = AutoModelForMaskedLM.from_pretrained(base_model_name)  # type: ignore
            peft_model = PeftModel.from_pretrained(base_model, model_path)  # type: ignore
            print("🔄 Merging PEFT weights into base model to avoid compatibility issues...")
            model = peft_model.merge_and_unload()  # type: ignore
            print("✅ Successfully merged PEFT model weights - now using as regular model")

        elif model_path:
            print(f"Loading MLM model from local path: {model_path}")
            model = AutoModelForMaskedLM.from_pretrained(model_path)  # type: ignore
        else:
            print(f"Loading MLM model from Hugging Face: {spec['hf_id']}")
            model = AutoModelForMaskedLM.from_pretrained(spec['hf_id'])  # type: ignore

        # Check if dataset already has labels (pre-masked from ray_datawriter.py)
        if "labels" in ds["train"].column_names:  # type: ignore
            print("✅ Dataset already contains pre-masked labels from ray_datawriter.py")
            print("   Using DataCollatorWithPadding (no additional masking needed)")
            collator = DataCollatorWithPadding(tokenizer=tokenizer)  # type: ignore
        else:
            # Dataset not pre-masked, apply masking during training
            if masking_mode != "default":
                print(f"Using CustomMaskingDataCollator with mode: {masking_mode}")
                collator = CustomMaskingDataCollator(
                    tokenizer=tokenizer,
                    mode=masking_mode,
                    mlm_probability=config.get("mlm_prob", 0.15)
                )
            else:
                print("Using DataCollatorForLanguageModeling for standard MLM masking.")
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

        # Check if we already have a PEFT model
        if hasattr(model, 'peft_config'):
            print("⚠️  Model already has PEFT adapters. Adding new LoRA layers on top.")
            # For existing PEFT models, we can add additional adapters
            # This creates a new adapter on top of the existing one
            print("Note: This will create stacked adapters. Consider merging previous adapters first if needed.")

        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=config.get("lora_target_modules", ["query", "value"]),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias="none",
        )

        # Apply new LoRA configuration
        if not hasattr(model, 'peft_config'):
            # Fresh model - apply LoRA normally
            model = get_peft_model(model, lora_config)  # type: ignore
        else:
            # Existing PEFT model - add new adapter
            from peft import get_peft_model
            model = get_peft_model(model, lora_config)  # type: ignore

        # LoRA will freeze many weights, but ensure the MLM head is trainable
        for name, param in model.named_parameters():  # type: ignore
            if "cls" in name or "LMPredictionHead" in name:
                param.requires_grad = True
        model.print_trainable_parameters()  # type: ignore
        lora_applied = True
    elif hasattr(model, 'peft_config'):
        print("✅ Using existing PEFT model without additional LoRA layers")
        lora_applied = True
    else:
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
        per_device_eval_batch_size=config.get("eval_batch_size", max(1, config["batch_size"] // 2)),
        learning_rate=config.get("learning_rate", 5e-5),
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
        # Fix for num_items_in_batch compatibility issue
        include_num_input_tokens_seen=False,
        # Memory management for evaluation
        dataloader_pin_memory=False,
        eval_accumulation_steps=config.get("eval_accumulation_steps", 1),
        # Gradient stability improvements
        max_grad_norm=config.get("max_grad_norm", 1.0),
        warmup_steps=config.get("warmup_steps", 500),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        # Better scheduler
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
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
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes (head + workers)")
    parser.add_argument("--num_gpu_per_node", type=int, default=1, help="Number of GPUs per node")
    parser.add_argument("--s3_output_path", type=str, default=None, help="S3 path to sync best model after training (optional)")
    parser.add_argument("--sample_rate", type=float, default=None, help="Sample rate for train/val splits (0 < rate < 1)")
    parser.add_argument("--masking_mode", type=str, default="default",
                       choices=["default", "tra", "trb", "tcr_pairing", "mhc_binding", "antigenic_specificity"],
                       help="Masking mode: default (standard MLM), tra/trb (preserve first/last 4 AA), others (mask second sequence)")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Local path to model directory (overrides model_key from config)")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model name/path for PEFT adapters (if model_path is a PEFT adapter)")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    config["test"] = args.test
    if args.s3_output_path:
        config["s3_output_path"] = args.s3_output_path
    if args.sample_rate:
        config["sample_rate"] = args.sample_rate
    if args.model_path:
        config["model_path"] = args.model_path
    if args.base_model:
        config["base_model"] = args.base_model
    config["masking_mode"] = args.masking_mode

    trainer = TorchTrainer(
        train_func,  # type: ignore
        scaling_config=ScalingConfig(
            num_workers=args.num_nodes,
            use_gpu=True,
            resources_per_worker={"GPU": args.num_gpu_per_node},
        ),
        run_config=RunConfig(
            failure_config=FailureConfig(max_failures=-1),
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
        train_loop_config=config,
    )  # type: ignore
    result = trainer.fit()  # type: ignore
    print(f"Ray DDP training finished with {args.num_nodes} nodes × {args.num_gpu_per_node} GPUs = {args.num_nodes * args.num_gpu_per_node} total GPUs!", result)

if __name__ == "__main__":
    main() 