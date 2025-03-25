import os
import sys
import json
import wandb
import torch
import evaluate
import torchvision
import pandas as pd
import plotly.express as px
import torch.distributed as dist

from tqdm import tqdm
from functools import partial

from collections import deque
from datetime import timedelta
from tokenizers import Tokenizer
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from quest.utils import collate_fn, collate_tf
from quest.constants import PAD_TOKEN_ID
from quest.models.factory import get_model 
from quest.dataset import AminoAcidDataset

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        """
        patience (int): How many epochs to wait after last improvement.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        # Check if val_loss improved sufficiently.
        if self.best_score - val_loss > self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Trainer:
    def __init__(self, config, sweep=False):
        """Initialize the Trainer, including setting up the distributed environment."""
        self.sweep = sweep
        self.is_distributed = "LOCAL_RANK" in os.environ

        if self.is_distributed:
            # Distributed Training (DDP) Setup
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])

            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=10))

            if self._is_main_process():
                # Rank 0 initializes wandb
                wandb.init(project="quest", config=config)
                config_dict = dict(wandb.config)  # Convert to standard dictionary
            else:
                config_dict = {}  
            
            # Broadcast config safely to all processes
            self.config = self._broadcast_config(config_dict)
        else:
            # Single-GPU / CPU Mode
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1

            if self.sweep:
                wandb.init(project="quest_sweep", config=config)
                self.config = dict(wandb.config)
            else:
                self.config = config

        # Load dataset 
        self.dataset = self.config["dataset"]
        self.train_dataset, self.val_dataset, self.test_dataset= self._load_preprocessed_dataset()
        self.pad_id = self.tokenizer.pad_token_id or 0
        if self.config["model_type"] == "transformer":
            self.collate_fn = partial(collate_tf, pad_token_id=self.pad_id)
        else:
            self.collate_fn = partial(collate_fn, pad_token_id=self.pad_id)

        self._prepare_dataloaders()
        self._initialize_model()

        # Initialize metrics
        self.hf_train_accuracy = evaluate.load("accuracy")
        self.hf_accuracy = evaluate.load("accuracy")
        self.hf_bleu = evaluate.load("bleu")
        self.hf_rouge = evaluate.load("rouge")
        self.hf_meteor = evaluate.load("meteor")
        self.hf_perplexity = evaluate.load("perplexity")
        
        if self._is_main_process():
            print(f"Trainer initialized. RANK={self.global_rank}, WORLD_SIZE={self.world_size}")


    def _save_checkpoint(self, epoch, best_val_loss):
        """Saves model/optimizer states to file (only in main process)."""
        if not self._is_main_process():
            return  # Only rank 0 saves to disk

        # If we're running a sweep, return immediately and do nothing.
        if self.sweep:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, self.config["checkpoint_path"])
        print(f"[Checkpoint] Saved at epoch {epoch} (val_loss={best_val_loss:.4f}) -> {self.config['checkpoint_path']}")

    def _load_checkpoint(self):
        """Loads model/optimizer states from file, returns (start_epoch, best_val_loss)."""
        ckpt_path = self.config["checkpoint_path"]
        if not os.path.exists(ckpt_path):
            print(f"[Checkpoint] No file found at {ckpt_path}, training from scratch.")
            return 0, float('inf')

        map_location = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

        print(f"[Checkpoint] Loaded from {ckpt_path}, resuming at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        return start_epoch, best_val_loss


    def _decode_ids(self, id_list):
        sequence = self.tokenizer.decode(id_list, special_tokens=False)
        return "".join(sequence)

    def _broadcast_config(self, config):
        """
        Broadcasts the config from rank 0 to all other ranks in DDP mode.
        If running in single-GPU mode, it simply returns the input config.
        """
        if not self.is_distributed:
            return config  # No need to broadcast in single-process mode

        # Convert config to JSON string
        config_str = json.dumps(dict(config))
        config_len = torch.tensor(len(config_str), dtype=torch.int, device=f"cuda:{self.local_rank}")

        # Broadcast length
        dist.broadcast(config_len, src=0)

        # Create buffer and broadcast config
        buffer = torch.empty(config_len.item(), dtype=torch.uint8, device=f"cuda:{self.local_rank}")
        if self.global_rank == 0:
            buffer[:] = torch.tensor(list(config_str.encode()), dtype=torch.uint8)

        dist.broadcast(buffer, src=0)
        config_str = "".join(map(chr, buffer.cpu().tolist()))

        return json.loads(config_str)  # Convert back to dict
    
    def _load_preprocessed_dataset(self):
        """
        Load the Hugging Face DatasetDict from disk and load the token dictionaries 
        (token_to_idx and idx_to_token) from Parquet files in the same dataset folder.
        Assumes self.config["dataset"] is a path to the folder where the HF dataset was saved,
        and that this folder contains "token_to_idx.parquet" and "idx_to_token.parquet".
        """
        

        # Load the HF DatasetDict
        ds_dict = load_from_disk(self.config["dataset"])
        train_ds = ds_dict["train"]
        val_ds = ds_dict["validation"]
        test_ds = ds_dict["test"]

        # Construct the paths to the token mapping files
        tokenizer_path = self.config['tokenizer_path']
        raw_tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer, 
                                            eos_token="[END]",
                                            unk_token="[UNK]",
                                            pad_token="[PAD]")
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        vocab_dict = tokenizer_data['model']['vocab']
        idx_to_token = {v: k for k, v in vocab_dict.items()}
        #dataset_folder = self.config["dataset"]
        self.tokenizer = tokenizer
        #self.padding_length = padding_length
        # Return the splits and dictionaries.
        return train_ds, val_ds, test_ds


    def _prepare_dataloaders(self):
        """
        Prepare DataLoaders from Hugging Face datasets. First, convert each dataset
        to PyTorch format by setting the format and specifying the column(s) to convert.
        Then use the standard PyTorch DataLoader with your collate_fn.
        """
        # Convert the HF datasets to torch format. We assume each dataset has a column "ids".
        self.train_dataset = self.train_dataset.with_format("torch", columns=['input_ids', 'target_ids'])
        self.val_dataset = self.val_dataset.with_format("torch", columns=['input_ids', 'target_ids'])
        self.test_dataset = self.test_dataset.with_format("torch", columns=['input_ids', 'target_ids'])

        # Set up samplers: use DistributedSampler if running in distributed mode,
        # otherwise use RandomSampler.
        if self.is_distributed:
            train_sampler = DistributedSampler(
                self.train_dataset, 
                num_replicas=self.world_size, 
                rank=self.global_rank, 
                shuffle=True, 
                drop_last=True)
        else:
            train_sampler = RandomSampler(self.train_dataset)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=self.collate_fn
        )

        if self.is_distributed:
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
                drop_last=True)
        else:
            val_sampler = RandomSampler(self.val_dataset)

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=self.collate_fn
        )

        if self.is_distributed:
            test_sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=self.world_size, 
                rank=self.global_rank, 
                shuffle=True, 
                drop_last=True)
        else:
            test_sampler = RandomSampler(self.test_dataset)

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            sampler=test_sampler,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=self.collate_fn
        )

    def _initialize_model(self):
        """Initialize model and optimizer."""
        #vocab_size = len({token for example in self.train_dataset for token in example["ids"]})
        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        vocab_size = self.tokenizer.vocab_size
        pad_token_id = (
            self.tokenizer.pad_token_id 
            if self.tokenizer.pad_token_id is not None 
            else 0  # or your fallback ID
        )
        print(f"local rank: {self.local_rank}")
        if self.config['model_type'] == "transformer":
            
            self.model = get_model(
            self.config["model_type"], vocab_size, self.config["embedding_dim"],
            self.config["hidden_dim"], self.config["num_layers"],
            self.config["nhead"], self.config["dropout"]
        ).to(device)    
        else:
            self.model = get_model(
                self.config["model_type"], vocab_size, self.config["embedding_dim"],
                self.config["hidden_dim"], self.config["num_layers"]
            ).to(device)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    def train(self):
        """Train and validate the model for multiple epochs with early stopping + checkpointing."""
        start_epoch = 0
        best_val_loss = float('inf')

        # ---------------- 1) Optional Resume from Checkpoint  ----------------
        if self.config.get("resume_from_checkpoint", False) and not self.sweep:
            start_epoch, best_val_loss = self._load_checkpoint()

        # ---------------- 2) EarlyStopping Instance  ----------------
        #  Only instantiate EarlyStopping if self.config["early_stop"] is True.
        use_early_stop = self.config.get("early_stop", False)
        if use_early_stop:
            early_stopper = EarlyStopping(
                patience=self.config.get("early_stopping_patience", 5),
                min_delta=1e-4
            )
            early_stopper.best_score = best_val_loss

        # ---------------- 3) Main Training Loop  ----------------
        num_epochs = self.config["num_epochs"]
        for epoch in range(start_epoch, num_epochs):
            self._train_one_epoch(epoch)

            # => Validation returns the average loss so we can track improvement
            val_loss = self._validate_one_epoch(epoch)

            # ---------------- 4) Checkpointing if val_loss improved  ----------------
            if not self.sweep and val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, best_val_loss)

            # ---------------- 5) Early Stopping Check  ----------------
            if use_early_stop:
                early_stopper(val_loss)
                if early_stopper.early_stop:
                    if self._is_main_process():
                        print(f"[EarlyStopping] Stopping early at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
                    break

        # Commented out test loop for now
        #self._test_model()
    def _is_main_process(self):
        return (not self.is_distributed) or (self.is_distributed and self.global_rank == 0)

    def _log_metrics(self, total_loss, total_correct, total_samples, stage, epoch, additional_metrics=None):
        """
        Logs training/validation metrics. Supports both single-GPU (wandb sweeps) and multi-GPU (torchrun).
        """
        if torch.cuda.is_available():
            device = f"cuda:{self.local_rank}"
        else:
            device = "cpu"

        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_correct_tensor = torch.tensor(total_correct, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)

        if self.is_distributed:
            # Reduce metrics across all GPUs
            dist.reduce(total_loss_tensor, dst=0)
            dist.reduce(total_correct_tensor, dst=0)
            dist.reduce(total_samples_tensor, dst=0)

        if self._is_main_process():
            # Avoid divide by zero
            if total_samples_tensor.item() > 0:
                avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
                avg_accuracy = total_correct_tensor.item() / total_samples_tensor.item()
                perplexity = torch.exp(torch.tensor(avg_loss))
            else:
                avg_loss = float("inf")
                avg_accuracy = 0.0
                perplexity = float("inf")
            log_data = {f"{stage}_loss": avg_loss, 
                        f"{stage}_accuracy": avg_accuracy,
                        f"{stage}_perplexity": perplexity}
            if additional_metrics is not None:
                log_data.update(additional_metrics)
            if epoch is not None:
                log_data["epoch"] = epoch

            wandb.log(log_data)


    def _train_one_epoch(self, epoch):
        """Train model for one epoch using full-sequence predictions."""
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1} - Training") if self._is_main_process() else None

        scaler = torch.amp.GradScaler()
        for inputs, targets, attention_mask in self.train_loader:
            inputs, targets = inputs.to(self.local_rank), targets.to(self.local_rank)
            attention_mask = attention_mask.to(self.local_rank)
            # outputs: (batch, seq_length, vocab_size)
            
            outputs = self.model(inputs, attention_mask=attention_mask)
            batch_size, seq_length, vocab_size = outputs.size()

            # Flatten outputs and targets for loss computation
            outputs_flat = outputs.view(-1, vocab_size)          # shape: (batch * seq_length, vocab_size)
            targets_flat = targets.view(-1)                        # shape: (batch * seq_length)

            # Compute loss over all tokens.
            loss = self.criterion(outputs_flat, targets_flat)
            # If the criterion uses "mean", multiply by the number of tokens to get the total loss:
            loss = loss * (batch_size * seq_length)

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()
            # Compute token-level accuracy.
            predicted = torch.argmax(outputs_flat, dim=-1)
            total_correct += (predicted == targets_flat).sum().item()
            total_samples += targets_flat.size(0)

            ### (C) Hugging Face metric calculation
            pad_id = self.tokenizer.pad_token_id or 0
            mask = (targets_flat != pad_id)
            masked_preds = predicted[mask].cpu().numpy()
            masked_refs  = targets_flat[mask].cpu().numpy()

            # Add to HF evaluate metric
            self.hf_train_accuracy.add_batch(
                predictions=masked_preds,
                references=masked_refs
            )
        
            if self._is_main_process():
                # Optionally log the average loss per token for the current batch.
                pbar.set_postfix({"Loss": loss.item() / (batch_size * seq_length)})
                pbar.update(1)

        if self._is_main_process() and pbar is not None:
            pbar.close()

         # ADDED: compute HF accuracy
        hf_acc_dict = self.hf_train_accuracy.compute()
        hf_accuracy_value = hf_acc_dict["accuracy"]

        # Now log both if you want
        additional_metrics = {
            "hf_train_accuracy": hf_accuracy_value
        }

        self._log_metrics(total_loss, total_correct, total_samples, "train", epoch, additional_metrics)


    def _validate_one_epoch(self, epoch):
        """
        Validate model on the entire val_loader using full-sequence predictions
        and compute teacher-forced accuracy. Additionally, for each batch, decode
        the last ten sequences (inputs, targets, and predictions) and push them to a buffer.
        At the end of validation, write the buffer to file for review.
        """
        self.model.eval()
        total_loss, total_correct, total_tokens = 0.0, 0, 0
        

        device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"


        # self.hf_accuracy = evaluate.load("accuracy")  ### ADDED OR RE-INIT HERE
        # self.hf_bleu = evaluate.load("bleu")
        # self.hf_rouge = evaluate.load("rouge")
        # self.hf_meteor = evaluate.load("meteor")
        # self.hf_perplexity = evaluate.load("perplexity")

        # Use a list to accumulate decoded predictions from every batch.
        pred_buffer = []

        pbar = tqdm(total=len(self.val_loader), desc=f"Epoch {epoch+1} - Validation") if self._is_main_process() else None

        with torch.no_grad():
            all_position_accuracies = []
            for batch_idx, (inputs, targets, attention_mask) in enumerate(self.val_loader):
                # Move to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                attention_mask = attention_mask.to(device)

                # Single forward pass
                outputs = self.model(inputs, attention_mask=attention_mask)
                batch_size, seq_length, vocab_size = outputs.size()
                predicted = torch.argmax(outputs, dim=-1) 
                ### (A) Standard loss & accuracy calculation
                outputs_flat = outputs.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                loss = self.criterion(outputs_flat, targets_flat)
                total_loss += loss.item() * (batch_size * seq_length)
                predicted_flat = torch.argmax(outputs_flat, dim=-1)
                total_correct += (predicted_flat == targets_flat).sum().item()
                total_tokens += batch_size * seq_length

                ### (C) Hugging Face metric calculation
                pad_id = self.tokenizer.pad_token_id or 0
                mask = (targets_flat != pad_id)
                masked_preds = predicted_flat[mask].cpu().numpy()
                masked_refs  = targets_flat[mask].cpu().numpy()

                # Add to HF evaluate metric
                self.hf_accuracy.add_batch(
                    predictions=masked_preds,
                    references=masked_refs
                )

                ### (D) For BLEU/ROUGE/METEOR, decode each sequence
                batch_pred_texts = []
                batch_ref_texts = []
                for b in range(batch_size):
                    pred_ids = predicted[b].cpu().tolist()
                    ref_ids  = targets[b].cpu().tolist()
                    pred_str = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    ref_str  = self.tokenizer.decode(ref_ids, skip_special_tokens=True)
                    batch_pred_texts.append(pred_str)
                    batch_ref_texts.append(ref_str)

                # DAdd to each metric
                # BLEU expects "references" as a list-of-list-of-strings
                # i.e. each prediction can have multiple references, so we do [[ref], [ref], ...]
                self.hf_bleu.add_batch(
                    predictions=batch_pred_texts,
                    references=[[r] for r in batch_ref_texts]
                )

                # ROUGE can accept single references as a list of strings
                self.hf_rouge.add_batch(
                    predictions=batch_pred_texts,
                    references=batch_ref_texts
                )

                self.hf_meteor.add_batch(
                    predictions=batch_pred_texts,
                    references=batch_ref_texts
                )
                ## (B) Position wise accuracy calculation
                position_accuracies = []
                for pos in range(seq_length):
                    # Calculate accuracy at position pos for the entire batch.
                    correct = (predicted[:, pos] == targets[:, pos]).sum().item()
                    acc = correct / batch_size
                    position_accuracies.append(acc)
                positions = list(range(1, seq_length + 1))
                all_position_accuracies.append(position_accuracies)
                ### (C) Decode first 20 sequences of the batch and push to buffer
                #if self._is_main_process():
                # predicted_sequences: [batch_size, seq_length]
                predicted_sequences = torch.argmax(outputs, dim=2)
                for b in range(min(20, batch_size)):
                    inpt_ids = inputs[b].cpu().tolist()
                    gold_ids = targets[b].cpu().tolist()
                    pred_ids = predicted_sequences[b].cpu().tolist()

                    # Decode using your _decode_ids method
                    input_str = self._decode_ids(inpt_ids)
                    gold_str = self._decode_ids(gold_ids)
                    pred_str = self._decode_ids(pred_ids)

                    pred_buffer.append((f"Batch {batch_idx}, Seq {b}", input_str, gold_str, pred_str))
                        
                if self._is_main_process():
                    pbar.set_postfix({"Loss": loss.item()})
                    pbar.update(1)

        if self._is_main_process() and pbar is not None:
            pbar.close()

        # ADDED: Compute the HF accuracy metric
        hf_accuracy_dict = self.hf_accuracy.compute()
        bleu_score   = self.hf_bleu.compute()
        rouge_score  = self.hf_rouge.compute()
        meteor_score = self.hf_meteor.compute()

        hf_accuracy_value = hf_accuracy_dict["accuracy"]
        hf_bleu_value = bleu_score["bleu"]
        hf_rouge_value = rouge_score["rougeL"]
        hf_meteor_value = meteor_score["meteor"]        


        # After validation, write the buffer to a file.
        #if self._is_main_process():
        # Create a wandb Table with appropriate column names.
        pred_table = wandb.Table(columns=["Epoch", "Identifier", "Input", "Target", "Prediction"])
        for ident, input_str, gold_str, pred_str in pred_buffer:
            pred_table.add_data(epoch, ident, input_str, gold_str, pred_str)
        #     #    wandb.log({"val_predictions_table": pred_table})

        # Compute average loss & accuracy
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        
        additional_metrics = {"prediction_table": pred_table,
                              "hf_val_accuracy": hf_accuracy_value,
                              "hf_val_bleu": hf_bleu_value,
                              "hf_val_rouge": hf_rouge_value,
                              "hf_val_meteor": hf_meteor_value}
        
        self._log_metrics(total_loss, total_correct, total_tokens, "val", epoch,
                        additional_metrics)
        return avg_loss


    

        