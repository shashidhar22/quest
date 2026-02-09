#!/usr/bin/env python3
"""
TCR Alpha-Beta Pairing with Robust Contrastive Learning (V3 - Production Ready).

A complete implementation combining all fixes and best practices:

Key Features:
- Momentum encoder for stable distractor embeddings (MoCo-style)
- Sequence-based hidden positive masking (CPU-offloaded to DataLoader workers)
- Full-corpus retrieval evaluation (realistic metrics)
- Cross-GPU negative gathering with correct rank-offset indexing
- Proper temperature parameterization (sigmoid bounds, always has gradients)
- Deep projection heads (3-layer, following SimCLR)
- Attention-weighted pooling (better for ESM2 than CLS)
- Curriculum learning (short→long sequences)
- GradScaler, cosine LR scheduler with warmup, early stopping
- Checkpoint management with best model tracking
- Thread-safe caching with size limits
- Comprehensive logging via wandb

Usage:
    torchrun --nproc_per_node=4 tcr_contrastive_v3.py \
        --data_path data/deduplicated/full/foundation_permutations/ \
        --output_dir ./output/tcr_v3 \
        --paired_permutation_keys tra_trb \
        --alpha_distractor_keys tra \
        --beta_distractor_keys trb

Author: Refactored with fixes from V1, V2, V2.1
"""

import argparse
import glob
import os
import threading
import warnings
from collections import OrderedDict
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from quest.data.datasets import TCRPairingDataset
from quest.metrics.sequence_metrics import SequenceSimilarityCalculator
from quest.models.attention_pooling import AttentionPooling, DeepProjectionHead
from quest.models.contrastive_loss import DistributedRobustInfoNCE, gather_embeddings
from quest.models.dual_encoder import TCRDualEncoder, MomentumEncoder, DistractorManager
from quest.models.lora_utils import apply_lora_to_encoder
from quest.training.callbacks import EarlyStopping
from quest.training.collators import ContrastivePairCollator
from quest.training.samplers import CurriculumSampler, DistributedCurriculumSampler

# =============================================================================
# Hardware Optimizations
# =============================================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# Thread-Safe Caching (Fix #11, #12)
# =============================================================================

class ThreadSafeCache:
    """
    Thread-safe LRU cache with size limits.
    Prevents memory leaks in multi-worker DataLoader scenarios.
    """
    
    def __init__(self, max_size: int = 3):
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[any]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    def set(self, key: str, value: any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = value
    
    def clear(self):
        with self._lock:
            self._cache.clear()


# Global caches
PAIRED_DATA_CACHE = ThreadSafeCache(max_size=3)
UNPAIRED_DATA_CACHE = ThreadSafeCache(max_size=3)

# Global debug logger
DEBUG_LOG_FILE = None


def debug_log(msg: str, also_print: bool = True):
    """Write debug message to log file and optionally to stdout."""
    if also_print:
        print(msg)
    if DEBUG_LOG_FILE is not None:
        try:
            with open(DEBUG_LOG_FILE, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"Warning: Could not write to debug log: {e}")



# =============================================================================
# Full-Corpus Evaluator (Fix #3, #9)
# =============================================================================

class FullCorpusEvaluator:
    """
    Evaluate with full-corpus retrieval.
    
    Unlike in-batch evaluation (which inflates metrics), this computes
    Recall@K against the ENTIRE validation set, giving realistic metrics.
    
    Expected metric ranges:
    - In-batch Recall@10 with batch_size=64: ~95%
    - Full-corpus Recall@10 with 10K candidates: ~15-40%
    """
    
    def __init__(self, k_values: List[int] = [1, 5, 10, 20, 100]):
        self.k_values = k_values
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataset: TCRPairingDataset,
        tokenizer,
        device: torch.device,
        batch_size: int = 128,
        max_length: int = 320,
    ) -> Dict[str, float]:
        model.eval()
        
        base_model = model.module if hasattr(model, 'module') else model
        
        # Encode all sequences
        def encode_all(sequences: List[str]) -> torch.Tensor:
            embeddings = []
            
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i + batch_size]
                
                encoded = tokenizer(
                    batch_seqs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    emb = base_model.encode(
                        encoded["input_ids"].to(device),
                        encoded["attention_mask"].to(device),
                    )
                
                embeddings.append(emb.cpu())
            
            return torch.cat(embeddings, dim=0)
        
        # Encode all alphas and betas
        alpha_emb = encode_all(dataset.alpha_seqs).to(device)
        beta_emb = encode_all(dataset.beta_seqs).to(device)
        
        # Compute ranks in chunks (avoid OOM)
        num_samples = len(dataset)
        all_ranks = []
        chunk_size = 1000
        
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            
            # Similarity: (chunk, all_betas)
            sim = alpha_emb[start:end] @ beta_emb.T
            
            # Sort and find rank of true positive
            sorted_idx = sim.argsort(dim=1, descending=True)
            
            for i, global_idx in enumerate(range(start, end)):
                rank = (sorted_idx[i] == global_idx).nonzero(as_tuple=True)[0]
                if len(rank) > 0:
                    all_ranks.append(rank[0].item() + 1)  # 1-indexed
        
        if not all_ranks:
            return {}
        
        ranks = np.array(all_ranks)
        
        metrics = {
            "mrr": float(np.mean(1.0 / ranks)),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
        }
        
        for k in self.k_values:
            metrics[f"recall@{k}"] = float(np.mean(ranks <= k))
        
        return metrics



# =============================================================================
# Trainer
# =============================================================================

class TCRContrastiveTrainer:
    """
    Complete trainer with all fixes applied.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.is_distributed = "LOCAL_RANK" in os.environ

        # Setup debug logging
        global DEBUG_LOG_FILE
        if config.get("debug_log_file"):
            DEBUG_LOG_FILE = config["debug_log_file"]
        else:
            # Default to output_dir/debug.log
            DEBUG_LOG_FILE = os.path.join(config["output_dir"], "debug.log")

        # Create output dir and initialize log file
        os.makedirs(config["output_dir"], exist_ok=True)
        if not self.is_distributed or int(os.environ.get("RANK", 0)) == 0:
            with open(DEBUG_LOG_FILE, "w") as f:
                f.write(f"Debug log started at {os.popen('date').read().strip()}\n")
                f.write(f"Config: {config}\n")
                f.write("=" * 60 + "\n\n")
            print(f"Debug log: {DEBUG_LOG_FILE}")

        # Setup distributed
        if self.is_distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=30),
            )
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_model()
        self._setup_data()
        self._setup_training()
        
        # Wandb
        self.use_wandb = (
            config.get("report_to") == "wandb" 
            and WANDB_AVAILABLE 
            and self._is_main()
        )
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "tcr-pairing-v3"),
                name=config.get("wandb_run_name"),
                config=config,
            )
    
    def _is_main(self) -> bool:
        return self.global_rank == 0
    
    def _setup_model(self):
        if self._is_main():
            print(f"Loading model: {self.config['model_name']}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Temperature: use user-provided value, or 0.5 for overfit check, or 0.07 default
        user_temp = self.config.get("temperature", None)
        if user_temp is not None and user_temp != 0.07:  # User explicitly set temperature
            initial_temp = user_temp
        elif self.config.get("overfit_check", False):
            initial_temp = 0.5  # Softer distribution for learning
        else:
            initial_temp = 0.07

        self.model = TCRDualEncoder(
            model_name=self.config["model_name"],
            projection_dim=self.config.get("projection_dim", 256),
            pooling=self.config.get("pooling", "attention"),
            initial_temperature=initial_temp,
            projection_layers=self.config.get("projection_layers", 3),
            use_batchnorm=self.config.get("use_batchnorm", False),
        )

        if self.config.get("use_lora", True):
            self.model = apply_lora_to_encoder(self.model, self.config)
            if self._is_main():
                self.model.encoder.print_trainable_parameters()
        else:
            # Freeze encoder when LoRA is disabled
            for param in self.model.encoder.parameters():
                param.requires_grad = False

            # Optionally unfreeze the last N transformer layers
            unfreeze_layers = self.config.get("unfreeze_layers", 0)
            if unfreeze_layers > 0:
                # ESM2 structure: model.encoder.layer[i] (from HuggingFace transformers)
                encoder = self.model.encoder
                num_layers = len(encoder.encoder.layer)

                if unfreeze_layers > num_layers:
                    if self._is_main():
                        print(f"Warning: unfreeze_layers={unfreeze_layers} > num_layers={num_layers}, unfreezing all")
                    unfreeze_layers = num_layers

                # Unfreeze last N layers
                for i in range(num_layers - unfreeze_layers, num_layers):
                    for param in encoder.encoder.layer[i].parameters():
                        param.requires_grad = True

                # Also unfreeze layer norm after transformer blocks
                if hasattr(encoder.encoder, 'emb_layer_norm_after'):
                    for param in encoder.encoder.emb_layer_norm_after.parameters():
                        param.requires_grad = True

                if self._is_main():
                    unfrozen = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
                    total = sum(p.numel() for p in self.model.encoder.parameters())
                    print(f"LoRA disabled - unfroze last {unfreeze_layers} encoder layers: {unfrozen:,}/{total:,} params")
            else:
                if self._is_main():
                    print("LoRA disabled - encoder frozen, training projection/pooler only")

        # Load pretrained PEFT checkpoint if provided
        pretrained_path = self.config.get("pretrained_checkpoint")
        if pretrained_path:
            if self._is_main():
                print(f"Loading pretrained checkpoint: {pretrained_path}")

            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)

            # The checkpoint has keys like: base_model.model.esm.encoder...
            # Our model.encoder has keys like: base_model.model.esm.encoder...
            # We need to load into model.encoder

            # Filter to only encoder keys and adjust prefix
            encoder_state = {}
            for k, v in state_dict.items():
                # Keep keys that are part of the ESM encoder
                if k.startswith("base_model.model.esm"):
                    encoder_state[k] = v

            if encoder_state:
                # Load with strict=False to allow missing projection head keys
                missing, unexpected = self.model.encoder.load_state_dict(
                    encoder_state, strict=False
                )
                if self._is_main():
                    print(f"  Loaded {len(encoder_state)} encoder weights")
                    if missing:
                        # Filter out expected missing keys (projection head, etc.)
                        real_missing = [m for m in missing if "lora" in m.lower()]
                        if real_missing:
                            print(f"  Missing LoRA keys: {len(real_missing)}")
                    if unexpected:
                        print(f"  Unexpected keys: {len(unexpected)}")
            else:
                if self._is_main():
                    print("  Warning: No encoder keys found in checkpoint!")

        # Enable gradient checkpointing (saves memory, required for 650M model)
        if self.config.get("gradient_checkpointing", True):
            self.model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.model = self.model.to(self.device)

        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=True,
            )
    
    def _setup_data(self):
        config = self.config

        # Fast mode for overfit check: skip distractors and limit samples
        is_overfit_check = config.get("overfit_check", False)
        skip_distractors = is_overfit_check

        # Use smaller batch size for overfit check (faster iterations)
        if is_overfit_check:
            config["batch_size"] = min(config.get("batch_size", 64), 32)  # Max 32 for overfit check

        # Need enough samples for at least 1 batch after train/val split (80% train)
        max_samples = config.get("batch_size", 64) * 4 if is_overfit_check else None

        if is_overfit_check and self._is_main():
            print("\n[Fast Mode] Overfit check enabled:")
            print(f"  - Skipping distractor loading")
            print(f"  - Using batch_size={config['batch_size']} (reduced for memory)")
            print(f"  - Limiting to {max_samples} samples\n")

        # Datasets
        self.train_dataset = TCRPairingDataset(
            data_path=config["data_path"],
            paired_permutation_keys=config.get("paired_permutation_keys", ["tra_trb"]),
            alpha_distractor_keys=config.get("alpha_distractor_keys", ["tra"]),
            beta_distractor_keys=config.get("beta_distractor_keys", ["trb"]),
            split="train",
            local_rank=self.local_rank,
            distractor_sample_size=config.get("distractor_sample_size", 500_000),
            skip_distractors=skip_distractors,
            max_samples=max_samples,
        )

        self.val_dataset = TCRPairingDataset(
            data_path=config["data_path"],
            paired_permutation_keys=config.get("paired_permutation_keys", ["tra_trb"]),
            split="val",
            local_rank=self.local_rank,
            max_samples=max_samples if is_overfit_check else None,
        )
        
        # Collator
        self.collator = ContrastivePairCollator(
            tokenizer=self.tokenizer,
            similarity_threshold=config.get("similarity_threshold", 0.9),
            max_length=config.get("max_length", 320),
            var_region_len=config.get("var_region_len", 150),
        )
        
        # Sampler - disable curriculum for overfit check (use all data)
        num_epochs = config.get("num_epochs", 10)
        use_curriculum = config.get("use_curriculum", True) and not is_overfit_check
        if use_curriculum:
            if self.is_distributed:
                self.train_sampler = DistributedCurriculumSampler(
                    self.train_dataset,
                    num_epochs=num_epochs,
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                )
            else:
                self.train_sampler = CurriculumSampler(
                    self.train_dataset,
                    num_epochs=num_epochs,
                )
        else:
            if self.is_distributed:
                # For overfit check with small dataset, don't drop_last
                self.train_sampler = DistributedSampler(
                    self.train_dataset, shuffle=True,
                    drop_last=not is_overfit_check
                )
            else:
                self.train_sampler = None
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.get("batch_size", 64),
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            num_workers=config.get("num_workers", 4) if not is_overfit_check else 0,
            prefetch_factor=2 if not is_overfit_check else None,
            pin_memory=True,
            persistent_workers=False,  # Fix #12
            drop_last=not is_overfit_check,  # Don't drop for overfit check
            collate_fn=self.collator,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.get("batch_size", 64),
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collator,
        )
        
        # Distractors
        if self.train_dataset.unpaired_alphas or self.train_dataset.unpaired_betas:
            self.distractor_manager = DistractorManager(
                model=self.model,
                tokenizer=self.tokenizer,
                unpaired_alphas=self.train_dataset.unpaired_alphas,
                unpaired_betas=self.train_dataset.unpaired_betas,
                device=self.device,
                initial_pool_size=config.get("initial_distractor_pool", 50_000),
                is_main=self._is_main(),
            )
        else:
            self.distractor_manager = None
            if self._is_main():
                warnings.warn("No distractors available!")
        
        # Evaluator
        self.evaluator = FullCorpusEvaluator()
    
    def _setup_training(self):
        config = self.config

        # Learning rate
        base_lr = config.get("learning_rate", 2e-4)
        if config.get("overfit_check", False):
            # Moderate LR for overfit check - training LoRA + projection/pooler
            # LoRA needs lower LR than randomly initialized projection head
            lr = config.get("overfit_lr") or 1e-3
            if self._is_main():
                print(f"Using overfit check LR: {lr} (LoRA + projection/pooler)")
        else:
            lr = base_lr

        # Optimizer with separate param groups for different learning rates
        base_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Group parameters: projection/pooler get higher LR than LoRA
        projection_params = []
        pooler_params = []
        lora_params = []
        other_params = []

        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            if 'projection' in name:
                projection_params.append(param)
            elif 'pooler' in name and 'encoder' not in name:  # Custom pooler, not ESM pooler
                pooler_params.append(param)
            elif 'lora' in name.lower():
                lora_params.append(param)
            else:
                other_params.append(param)

        # Projection/pooler get 10x LR (randomly initialized, need faster learning)
        param_groups = [
            {'params': projection_params, 'lr': lr * 10, 'name': 'projection'},
            {'params': pooler_params, 'lr': lr * 10, 'name': 'pooler'},
            {'params': lora_params, 'lr': lr, 'name': 'lora'},
            {'params': other_params, 'lr': lr, 'name': 'other'},
        ]

        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]

        if self._is_main():
            print(f"[Optimizer] Parameter groups:")
            for g in param_groups:
                total_params = sum(p.numel() for p in g['params'])
                print(f"  {g['name']}: {len(g['params'])} tensors, {total_params:,} params, lr={g['lr']:.6f}")

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,  # Default LR (overridden by param groups)
            weight_decay=config.get("weight_decay", 0.01),
        )
        
        # Scheduler (cosine with warmup)
        grad_accum = config.get("gradient_accumulation_steps", 1)
        num_epochs = config.get("num_epochs", 10)
        steps_per_epoch = len(self.train_loader) // grad_accum
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # GradScaler - NOT needed for bfloat16 (only for float16)
        # bfloat16 has larger dynamic range and doesn't need loss scaling
        self.scaler = None  # Disabled for bfloat16
        
        # Loss
        self.criterion = DistributedRobustInfoNCE(
            rank=self.global_rank,
            world_size=self.world_size,
            symmetric=config.get("symmetric_loss", True),
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 5),
            mode="max",
        ) if config.get("early_stopping", True) else None
        
        # State
        self.global_step = 0
        self.best_metric = float("-inf")
    
    def overfit_single_batch(self, num_steps: int = 100):
        """
        Debug utility: Try to overfit on a single batch.

        If the model is working correctly, loss should decrease toward 0.
        If loss doesn't decrease, there's a fundamental issue with:
        - Model architecture
        - Loss function
        - Optimizer setup
        - Gradient flow
        """
        if self._is_main():
            debug_log(f"\n{'='*60}")
            debug_log("SINGLE BATCH OVERFIT CHECK")
            debug_log(f"{'='*60}")
            debug_log(f"Running {num_steps} iterations on a single batch...")
            debug_log("Expected: Loss should decrease significantly (ideally toward 0)")
            debug_log(f"{'='*60}\n")

        self.model.train()

        # Train full model (LoRA + projection/pooler)
        # The encoder needs to learn which positions matter for TCR pairing.
        # With frozen encoder, embeddings are dominated by conserved framework regions.
        base_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Freeze temperature during overfit check to reduce instability
        if hasattr(base_model, 'temperature_logit'):
            base_model.temperature_logit.requires_grad = False

        if self._is_main():
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            lora_params = sum(p.numel() for n, p in base_model.named_parameters()
                             if p.requires_grad and 'lora' in n.lower())
            debug_log(f"[Trainable] {trainable:,} params (LoRA: {lora_params:,}, projection/pooler: {trainable - lora_params:,})")

        # Check dataset size
        if self._is_main():
            debug_log(f"[Dataset Info]")
            debug_log(f"  Train dataset size: {len(self.train_dataset)}")
            debug_log(f"  Train loader batches: {len(self.train_loader)}")
            if len(self.train_dataset) == 0:
                debug_log("  ERROR: Train dataset is empty!")
                debug_log("  Check your data_path and paired_permutation_keys")
                return [], []
            if len(self.train_loader) == 0:
                debug_log("  ERROR: Train loader is empty!")
                debug_log(f"  Dataset has {len(self.train_dataset)} samples but batch_size={self.config.get('batch_size', 64)}")
                debug_log("  This can happen if dataset size < batch_size with drop_last=True")
                return [], []

        # Get a single batch
        batch_iter = iter(self.train_loader)
        try:
            batch = next(batch_iter)
        except StopIteration:
            if self._is_main():
                debug_log("ERROR: Could not get a batch from DataLoader!")
                debug_log(f"  Dataset size: {len(self.train_dataset)}")
                debug_log(f"  Batch size: {self.config.get('batch_size', 64)}")
            return [], []

        # Unpack batch
        alpha_ids = batch["alpha_input_ids"].to(self.device)
        alpha_mask = batch["alpha_attention_mask"].to(self.device)
        beta_ids = batch["beta_input_ids"].to(self.device)
        beta_mask = batch["beta_attention_mask"].to(self.device)
        alpha_sim_mask = batch["alpha_sim_mask"]
        beta_sim_mask = batch["beta_sim_mask"]

        if self._is_main():
            debug_log(f"[Batch Info]")
            debug_log(f"  Alpha IDs shape: {alpha_ids.shape}")
            debug_log(f"  Beta IDs shape: {beta_ids.shape}")
            debug_log(f"  Alpha attention sum: {alpha_mask.sum().item()}")
            debug_log(f"  Beta attention sum: {beta_mask.sum().item()}")
            debug_log("")

        initial_loss = None
        losses = []
        accuracies = []

        for step in range(num_steps):
            self.optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward - disable re-centering for overfit test to simplify optimization
                outputs = self.model(alpha_ids, alpha_mask, beta_ids, beta_mask, recenter=False)

                alpha_local = outputs["alpha_embeddings"]
                beta_local = outputs["beta_embeddings"]
                temperature = outputs["temperature"]

                # Gather across GPUs (no-op if single GPU)
                alpha_all = gather_embeddings(alpha_local, self.world_size)
                beta_all = gather_embeddings(beta_local, self.world_size)

                # No distractors for overfit test (simpler)
                ext_alpha = torch.empty(0, alpha_local.shape[-1], device=self.device)
                ext_beta = torch.empty(0, beta_local.shape[-1], device=self.device)

                # Loss
                loss_dict = self.criterion(
                    alpha_local, beta_local,
                    alpha_sim_mask, beta_sim_mask,
                    alpha_all, beta_all,
                    ext_alpha, ext_beta,
                    temperature,
                )
                loss = loss_dict["loss"]

            # Backward - use gradient clipping to stabilize training
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            loss_val = loss_dict["loss"].item()
            acc_val = loss_dict["accuracy"].item()
            losses.append(loss_val)
            accuracies.append(acc_val)

            if initial_loss is None:
                initial_loss = loss_val

            if self._is_main() and (step % 10 == 0 or step == num_steps - 1):
                # Compute similarity stats to detect embedding collapse
                with torch.no_grad():
                    sim_matrix = torch.mm(alpha_local.float(), beta_local.float().t())
                    diag_sim = sim_matrix.diag().mean().item()  # matched pairs
                    off_diag_mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
                    off_diag_sim = sim_matrix[off_diag_mask].mean().item()  # non-matched pairs

                debug_log(f"Step {step:3d}: loss={loss_val:.4f}, acc={acc_val:.4f}, "
                          f"temp={temperature.item():.4f}, grad_norm={grad_norm:.4f}")
                debug_log(f"         sim(matched)={diag_sim:.4f}, sim(non-matched)={off_diag_sim:.4f}")

        # Summary
        if self._is_main():
            final_loss = losses[-1]
            final_acc = accuracies[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0

            debug_log(f"\n{'='*60}")
            debug_log("OVERFIT CHECK RESULTS")
            debug_log(f"{'='*60}")
            debug_log(f"Initial loss: {initial_loss:.4f}")
            debug_log(f"Final loss:   {final_loss:.4f}")
            debug_log(f"Loss reduction: {loss_reduction:.1f}%")
            debug_log(f"Initial accuracy: {accuracies[0]:.4f}")
            debug_log(f"Final accuracy:   {final_acc:.4f}")
            debug_log("")

            if final_loss < 0.1 and final_acc > 0.95:
                debug_log("SUCCESS: Model can overfit a single batch!")
                debug_log("  The training loop and model architecture are working.")
            elif loss_reduction > 50:
                debug_log("PARTIAL: Loss decreased but didn't fully converge.")
                debug_log("  Training is working but may need more steps or tuning.")
            elif loss_reduction > 10:
                debug_log("SLOW: Loss is decreasing slowly.")
                debug_log("  Check learning rate, optimizer, or gradient flow.")
            else:
                debug_log("FAILURE: Model cannot overfit a single batch!")
                debug_log("  Possible issues:")
                debug_log("  - Gradients not flowing (check requires_grad)")
                debug_log("  - Learning rate too low")
                debug_log("  - Loss function issue")
                debug_log("  - Model architecture problem")

            # Always show diagnostics
            debug_log("\n[Additional Diagnostics]")
            base_model = self.model.module if hasattr(self.model, 'module') else self.model

            # Check trainable parameters by component
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            debug_log(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

            # Break down by component
            debug_log("\n  [Trainable params by component]")
            for name, param in base_model.named_parameters():
                if param.requires_grad:
                    debug_log(f"    {name}: {param.numel():,} (grad_fn={param.grad is not None})")

            # Check gradient magnitudes
            debug_log("\n  [Gradient magnitudes]")
            has_grad = False
            grad_stats = []
            for name, param in base_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.abs().mean().item()
                    if grad_norm > 0:
                        has_grad = True
                        grad_stats.append((name, grad_norm))

            # Show top 5 largest gradients
            grad_stats.sort(key=lambda x: x[1], reverse=True)
            for name, grad_norm in grad_stats[:5]:
                debug_log(f"    {name}: {grad_norm:.6f}")
            if not grad_stats:
                debug_log("    No gradients found!")

            debug_log(f"\n  Gradients present: {has_grad}")

            # Check embedding similarity
            with torch.no_grad():
                outputs = self.model(alpha_ids, alpha_mask, beta_ids, beta_mask)
                alpha_emb = outputs["alpha_embeddings"]
                beta_emb = outputs["beta_embeddings"]

                # Cosine similarity of matched pairs
                cos_sim = (alpha_emb * beta_emb).sum(dim=1).mean().item()
                debug_log(f"  Avg cosine sim (matched pairs): {cos_sim:.4f}")

                # Check if embeddings are collapsed (all same)
                alpha_std = alpha_emb.std().item()
                beta_std = beta_emb.std().item()
                debug_log(f"  Alpha embedding std: {alpha_std:.6f}")
                debug_log(f"  Beta embedding std: {beta_std:.6f}")
                if alpha_std < 0.01 or beta_std < 0.01:
                    debug_log("  WARNING: Embeddings may be collapsed!")

            debug_log(f"{'='*60}\n")

        return losses, accuracies

    def train(self):
        config = self.config

        # Run overfit check if requested
        if config.get("overfit_check", False):
            self.overfit_single_batch(num_steps=config.get("overfit_steps", 100))
            return

        num_epochs = config.get("num_epochs", 10)
        grad_accum = config.get("gradient_accumulation_steps", 1)
        logging_steps = config.get("logging_steps", 50)
        eval_steps = config.get("eval_steps", 500)
        save_steps = config.get("save_steps", 500)
        refresh_steps = config.get("refresh_distractors_steps", 2000)
        num_distractors = config.get("num_distractors", 1000)
        
        if self._is_main():
            print(f"\n{'='*60}")
            print(f"Starting training for {num_epochs} epochs")
            print(f"  Batch size: {config.get('batch_size', 64)}")
            print(f"  Gradient accumulation: {grad_accum}")
            print(f"  World size: {self.world_size}")
            print(f"  Effective batch size: {config.get('batch_size', 64) * self.world_size * grad_accum}")
            print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            if hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)
            
            self._train_epoch(
                epoch, grad_accum, logging_steps, eval_steps, 
                save_steps, refresh_steps, num_distractors
            )
            
            # Epoch-end evaluation
            metrics = self._validate()
            if self._is_main():
                print(f"\nEpoch {epoch+1} - Recall@10: {metrics.get('recall@10', 0):.4f}, "
                      f"MRR: {metrics.get('mrr', 0):.4f}")
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(metrics.get("recall@10", 0)):
                    if self._is_main():
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self._save_final()
        
        if self.use_wandb:
            wandb.finish()
    
    def _train_epoch(
        self,
        epoch: int,
        grad_accum: int,
        logging_steps: int,
        eval_steps: int,
        save_steps: int,
        refresh_steps: int,
        num_distractors: int,
    ):
        self.model.train()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1}",
            disable=not self._is_main(),
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in pbar:
            # Unpack batch
            alpha_ids = batch["alpha_input_ids"].to(self.device)
            alpha_mask = batch["alpha_attention_mask"].to(self.device)
            beta_ids = batch["beta_input_ids"].to(self.device)
            beta_mask = batch["beta_attention_mask"].to(self.device)
            alpha_sim_mask = batch["alpha_sim_mask"]
            beta_sim_mask = batch["beta_sim_mask"]
            
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward
                outputs = self.model(alpha_ids, alpha_mask, beta_ids, beta_mask)
                
                alpha_local = outputs["alpha_embeddings"]
                beta_local = outputs["beta_embeddings"]
                temperature = outputs["temperature"]
                
                # Gather across GPUs
                alpha_all = gather_embeddings(alpha_local, self.world_size)
                beta_all = gather_embeddings(beta_local, self.world_size)
                
                # Sample distractors
                if self.distractor_manager:
                    ext_alpha, ext_beta = self.distractor_manager.sample(
                        num_distractors, num_distractors
                    )
                else:
                    ext_alpha = torch.empty(0, 256, device=self.device)
                    ext_beta = torch.empty(0, 256, device=self.device)
                
                # Loss
                loss_dict = self.criterion(
                    alpha_local, beta_local,
                    alpha_sim_mask, beta_sim_mask,
                    alpha_all, beta_all,
                    ext_alpha, ext_beta,
                    temperature,
                )
                loss = loss_dict["loss"] / grad_accum
            
            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss_dict["loss"].item()
            total_acc += loss_dict["accuracy"].item()
            num_batches += 1
            
            # Optimizer step
            if (step + 1) % grad_accum == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Update momentum encoder
                if self.distractor_manager:
                    self.distractor_manager.update_momentum(self.model)
                
                pbar.set_postfix({
                    "loss": f"{loss_dict['loss'].item():.4f}",
                    "acc": f"{loss_dict['accuracy'].item():.4f}",
                    "temp": f"{temperature.item():.4f}",
                })
                
                # Logging
                if self.global_step % logging_steps == 0 and self._is_main():
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": total_loss / num_batches,
                            "train/accuracy": total_acc / num_batches,
                            "train/temperature": temperature.item(),
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/step": self.global_step,
                        })
                
                # Refresh distractors
                if self.global_step % refresh_steps == 0 and self.distractor_manager:
                    self.distractor_manager.refresh()
                
                # Evaluation
                if self.global_step % eval_steps == 0:
                    metrics = self._validate()
                    if self._is_main():
                        print(f"\nStep {self.global_step} - Recall@10: {metrics.get('recall@10', 0):.4f}")
                        
                        if self.use_wandb:
                            wandb.log({f"eval/{k}": v for k, v in metrics.items()})
                        
                        if metrics.get("recall@10", 0) > self.best_metric:
                            self.best_metric = metrics["recall@10"]
                            self._save_checkpoint(is_best=True)
                    
                    self.model.train()
                
                # Checkpoint
                if self.global_step % save_steps == 0:
                    self._save_checkpoint()
        
        pbar.close()
    
    def _validate(self) -> Dict[str, float]:
        return self.evaluator.evaluate(
            self.model,
            self.val_dataset,
            self.tokenizer,
            self.device,
        )
    
    def _save_checkpoint(self, is_best: bool = False):
        if not self._is_main():
            return
        
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        model_to_save = self.model.module if self.is_distributed else self.model
        
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }
        
        path = os.path.join(output_dir, f"checkpoint-{self.global_step}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = os.path.join(output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model (recall@10={self.best_metric:.4f})")
        
        # Cleanup old checkpoints (keep 3)
        checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*.pt")))
        for old in checkpoints[:-3]:
            os.remove(old)
    
    def _save_final(self):
        if not self._is_main():
            return
        
        final_dir = os.path.join(self.config["output_dir"], "final")
        os.makedirs(final_dir, exist_ok=True)
        
        model_to_save = self.model.module if self.is_distributed else self.model
        
        torch.save(
            {"model_state_dict": model_to_save.state_dict(), "config": self.config},
            os.path.join(final_dir, "model.pt"),
        )
        self.tokenizer.save_pretrained(final_dir)
        
        print(f"\nFinal model saved to: {final_dir}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="TCR Alpha-Beta Pairing with Robust Contrastive Learning (V3)"
    )
    
    # Required
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Data
    parser.add_argument("--paired_permutation_keys", type=str, nargs="+", default=["tra_trb"])
    parser.add_argument("--alpha_distractor_keys", type=str, nargs="+", default=["tra"])
    parser.add_argument("--beta_distractor_keys", type=str, nargs="+", default=["trb"])
    parser.add_argument("--max_length", type=int, default=320)
    parser.add_argument("--distractor_sample_size", type=int, default=500_000)
    parser.add_argument("--similarity_threshold", type=float, default=0.9)
    parser.add_argument("--var_region_len", type=int, default=150,
                        help="Truncate sequences to this length before encoding (variable region only). "
                             "Set to 0 to disable truncation.")

    # Model
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                        help="Path to pretrained PEFT checkpoint (.pt file with model_state_dict)")
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--pooling", type=str, default="attention", choices=["attention", "cls", "mean"])
    parser.add_argument("--projection_layers", type=int, default=3)
    parser.add_argument("--use_batchnorm", action="store_true", default=False,
                        help="Use BatchNorm1d instead of LayerNorm in projection head. "
                             "BatchNorm can help break embedding collapse by normalizing across batch dimension.")
    parser.add_argument("--temperature", type=float, default=0.07)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora",
                        help="Disable LoRA (use frozen encoder for debugging)")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--unfreeze_layers", type=int, default=0,
                        help="Number of last encoder layers to unfreeze when LoRA is disabled. "
                             "ESM2-650M has 33 layers. Set to 0 to freeze all (projection only).")
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_distractors", type=int, default=1000)
    parser.add_argument("--initial_distractor_pool", type=int, default=50_000)
    
    # Curriculum & Features
    parser.add_argument("--use_curriculum", action="store_true", default=True)
    parser.add_argument("--no_curriculum", action="store_false", dest="use_curriculum")
    parser.add_argument("--symmetric_loss", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_amp", action="store_true", default=True)
    
    # Evaluation & Logging
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--refresh_distractors_steps", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Early Stopping
    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    
    # Wandb
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    parser.add_argument("--wandb_project", type=str, default="tcr-pairing-v3")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Debug / Overfit check
    parser.add_argument("--overfit_check", action="store_true", default=False,
                        help="Run single batch overfit check instead of full training")
    parser.add_argument("--overfit_steps", type=int, default=100,
                        help="Number of steps for overfit check")
    parser.add_argument("--overfit_lr", type=float, default=None,
                        help="Learning rate for overfit check (default: 10x normal lr)")
    parser.add_argument("--debug_log_file", type=str, default=None,
                        help="Path to write debug logs (default: output_dir/debug.log)")

    return parser.parse_args()


def main():
    args = parse_args()
    config = vars(args)

    trainer = TCRContrastiveTrainer(config)
    trainer.train()
    
    if trainer.is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()