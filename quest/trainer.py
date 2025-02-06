import os
import json
import pickle
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from quest.models.factory import get_model 
from quest.dataset import AminoAcidDataset 
from datetime import timedelta

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
        self.train_dataset, self.val_dataset, self.test_dataset = self._load_preprocessed_dataset()

        self._prepare_dataloaders()
        self._initialize_model()

        if self._is_main_process():
            print(f"Trainer initialized. RANK={self.global_rank}, WORLD_SIZE={self.world_size}")

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
        """Load dataset from a pickle file."""
        with open(self.dataset, "rb") as f:
            data = pickle.load(f)
        return data["train_data"], data["val_data"], data["test_data"]

    def _prepare_dataloaders(self):
        """Prepare distributed DataLoaders."""
        if self.is_distributed:
            train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=True, drop_last=True)
        else:
            train_sampler = RandomSampler(self.train_dataset)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2  
        )

        if self.is_distributed:
            val_sampler = DistributedSampler(self.val_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=True, drop_last=True)
        else:
            val_sampler = RandomSampler(self.val_dataset)

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2  
        )

        if self.is_distributed:
            test_sampler = DistributedSampler(self.test_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=True, drop_last=True)
        else:
            test_sampler = RandomSampler(self.test_dataset,)

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            sampler=test_sampler,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2  
        )

    def _initialize_model(self):
        """Initialize model and optimizer."""
        vocab_size = len(set(sum(self.train_dataset.sequences, [])))
        self.model = get_model(
            self.config["model_type"], vocab_size, self.config["embedding_dim"],
            self.config["hidden_dim"], self.config["num_layers"]
        ).to(self.local_rank)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    def train(self):
        """Train and validate the model for multiple epochs."""
        for epoch in range(self.config["num_epochs"]):
            self._train_one_epoch(epoch)
            self._validate_one_epoch(epoch)

        # Commented out test loop for now
        #self._test_model()
    def _is_main_process(self):
        return (not self.is_distributed) or (self.is_distributed and self.global_rank == 0)

    def _log_metrics(self, total_loss, total_correct, total_samples, stage, epoch):
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
            else:
                avg_loss = float("inf")
                avg_accuracy = 0.0
            
            log_data = {f"{stage}_loss": avg_loss, f"{stage}_accuracy": avg_accuracy}
            if epoch is not None:
                log_data["epoch"] = epoch

            wandb.log(log_data)


    def _train_one_epoch(self, epoch):
        """Train model for one epoch."""
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1} - Training") if self._is_main_process() else None

        scaler = torch.amp.GradScaler()
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.local_rank), targets.to(self.local_rank)
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(dim=1)
            total_correct += (predicted == targets).sum().item()
            total_samples += inputs.size(0)

            if self._is_main_process():
                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)

        if self._is_main_process() and pbar is not None:
            pbar.close()

        self._log_metrics(total_loss, total_correct, total_samples, "train", epoch)

    def _validate_one_epoch(self, epoch):
        """Validate model on validation dataset."""
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.local_rank), targets.to(self.local_rank)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                _, predicted = outputs.max(dim=1)

                total_loss += loss.item() * inputs.size(0)
                total_correct += (predicted == targets).sum().item()
                total_samples += inputs.size(0)

        self._log_metrics(total_loss, total_correct, total_samples, "val", epoch)
