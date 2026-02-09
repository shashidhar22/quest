#!/usr/bin/env python3
"""
TCR-Peptide-MHC Conditional Sequence Generation Trainer

Encoder-decoder architecture for conditional TCR sequence generation.
Uses ESM2 as encoder with a TransformerDecoder for autoregressive generation.

Architecture:
    [Encoder: ESM2]                              [Decoder: torch.nn.TransformerDecoder]
    [Beta <eos> Peptide <eos> MHC <eos>]  -->  Cross-Attention  -->  Alpha chain
             (context)                           + Causal LM            (target)

Features:
- ESM2 encoder with flash attention 2, frozen or with LoRA
- TransformerDecoder with cross-attention to encoder outputs
- Weight tying: decoder embeddings shared with ESM2 encoder embeddings
- Autoregressive next-token prediction with cross-entropy loss
- Support for ALPHA, BETA, and PEPTIDE generation tasks
- DDP for multi-GPU training
- Mixed precision (bfloat16)

Usage:
    python scripts/training/tcr_seq2seq_trainer.py \
        --data_path data/deduplicated/full/foundation_permutations/ \
        --output_dir ./output/tcr_seq2seq \
        --task ALPHA \
        --permutation_keys tra_trb_peptide_mhc_one tra_trb_peptide_mhc_one_mhc_two \
        --batch_size 16 \
        --gradient_accumulation_steps 8 \
        --num_epochs 10 \
        --use_lora \
        --overfit_check
"""

import argparse
import copy
import glob
import math
import os
import sys
from datetime import datetime, timedelta

# Add project root to path for direct script execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Backend abstraction for hardware-agnostic training
from quest.training.backends import AcceleratorBackend, get_backend
from quest.training.base_trainer import BaseTCRTrainer

# Optional: LoRA support
try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Optional: wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Optional: TCR stitching for full-length sequences
try:
    from quest.parsers.tcr_stitcher import TCRStitcher
    STITCHER_AVAILABLE = True
except ImportError:
    STITCHER_AVAILABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Extracted modules (Phase 4 refactor) ---
from quest.models.positional_encoding import PositionalEncoding
from quest.models.seq2seq_decoder import SelfAttnDropoutDecoderLayer, SelfAttnDropoutDecoder
from quest.models.seq2seq_model import TCRSeq2SeqModel, BLOSUM62SoftLoss
from quest.models.lora_utils import apply_lora_to_encoder, load_encoder_checkpoint
from quest.data.datasets import TCRSeq2SeqDataset, Seq2SeqCollator
from quest.data.permutation_utils import (
    GenerationTask, FIELDS, permutation_key_has_fields,
    get_required_fields_for_task, discover_permutation_keys,
)
from quest.data.generation_utils import greedy_decode, beam_search, sample_with_temperature
from quest.metrics.sequence_metrics import (
    levenshtein_distance, blosum62_similarity, sequence_identity,
    blosum62_normalized, BLOSUM62, STANDARD_AMINO_ACIDS,
)
from quest.metrics.biophysical import (
    compute_hydrophobicity_profile, hydrophobicity_correlation,
    compute_net_charge, aa_frequency_distribution, jensen_shannon_divergence,
    KYTE_DOOLITTLE, AMINO_ACID_CHARGE,
)
from quest.metrics.position_wise import PositionWiseAnalyzer
from quest.metrics.cdr_analysis import (
    CDRAnnotationCache, get_cdr_positions_anarci, build_cdr_position_weights,
    CDR_WEIGHT_PRESETS, ANARCI_AVAILABLE,
)

# Backward-compatible aliases for names used in inline code
AMINO_ACIDS = STANDARD_AMINO_ACIDS
AA_CHARGE = AMINO_ACID_CHARGE


# =============================================================================
# Sequence Alignment Utilities
# =============================================================================


# Optional: Biopython for alignment
try:
    from Bio import pairwise2
    from Bio.Align import substitution_matrices
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


def aligned_sequence_metrics(generated: str, reference: str) -> Dict[str, float]:
    """
    Compute metrics on properly aligned sequences using Biopython.

    Uses Needleman-Wunsch global alignment with BLOSUM62 scoring.

    Args:
        generated: Generated sequence
        reference: Reference sequence

    Returns:
        Dict with alignment-based metrics
    """
    if not BIOPYTHON_AVAILABLE:
        # Fallback to simple position-wise comparison
        return {
            "identity_aligned": sequence_identity(generated, reference),
            "blosum_score": blosum62_similarity(generated, reference),
            "alignment_available": False,
        }

    if not generated or not reference:
        return {
            "identity_aligned": 0.0,
            "blosum_score": 0.0,
            "alignment_available": True,
        }

    try:
        # Load BLOSUM62 for alignment scoring
        blosum62_matrix = substitution_matrices.load("BLOSUM62")

        # Global alignment (Needleman-Wunsch style)
        alignments = pairwise2.align.globalds(
            generated, reference,
            blosum62_matrix,
            -10,   # Gap open penalty
            -0.5,  # Gap extend penalty
        )

        if not alignments:
            return {
                "identity_aligned": 0.0,
                "blosum_score": 0.0,
                "alignment_available": True,
            }

        best_alignment = alignments[0]
        aligned_gen, aligned_ref, score, begin, end = best_alignment

        # Compute identity on aligned sequences
        matches = sum(1 for a, b in zip(aligned_gen, aligned_ref)
                      if a == b and a != '-')
        non_gap_positions = sum(1 for a, b in zip(aligned_gen, aligned_ref)
                                 if a != '-' or b != '-')
        identity = matches / max(non_gap_positions, 1)

        # Gap statistics
        gen_gaps = aligned_gen.count('-')
        ref_gaps = aligned_ref.count('-')

        return {
            "identity_aligned": identity,
            "blosum_score": score,
            "blosum_per_position": score / max(len(aligned_gen), 1),
            "alignment_length": len(aligned_gen),
            "gen_gaps": gen_gaps,
            "ref_gaps": ref_gaps,
            "alignment_available": True,
        }

    except Exception:
        return {
            "identity_aligned": sequence_identity(generated, reference),
            "blosum_score": blosum62_similarity(generated, reference),
            "alignment_available": False,
        }


# =============================================================================
# Evaluator: GenerationEvaluator
# =============================================================================


class GenerationEvaluator:
    """
    Evaluator for sequence generation with comprehensive biological metrics.

    Metrics:
    - Cross-entropy loss (teacher-forced)
    - Perplexity
    - Exact match rate
    - Length statistics
    - Sequence similarity (Levenshtein, identity, BLOSUM62)
    - Position-specific accuracy (CDR3 conserved positions)
    - Amino acid composition (hydrophobicity, charge, distribution)
    """

    def __init__(self, tokenizer, compute_detailed_metrics: bool = True):
        """
        Initialize the evaluator.

        Args:
            tokenizer: ESM2 tokenizer
            compute_detailed_metrics: Whether to compute advanced biological metrics
        """
        self.tokenizer = tokenizer
        self.compute_detailed_metrics = compute_detailed_metrics
        self.position_analyzer = PositionWiseAnalyzer() if compute_detailed_metrics else None

    def _compute_sequence_metrics(
        self,
        generated_seqs: List[str],
        reference_seqs: List[str],
    ) -> Dict[str, Any]:
        """
        Compute comprehensive sequence similarity and composition metrics.

        Args:
            generated_seqs: List of generated sequences
            reference_seqs: List of reference sequences

        Returns:
            Dict with sequence similarity, position, and composition metrics
        """
        metrics = {}

        # Clean sequences (remove spaces from tokenizer output)
        gen_clean = [s.replace(" ", "") for s in generated_seqs]
        ref_clean = [s.replace(" ", "") for s in reference_seqs]

        # ===== Sequence Similarity Metrics =====
        lev_distances = []
        identities = []
        blosum_scores = []
        blosum_norm_scores = []

        for gen, ref in zip(gen_clean, ref_clean):
            if gen and ref:
                lev_distances.append(levenshtein_distance(gen, ref))
                identities.append(sequence_identity(gen, ref))
                blosum_scores.append(blosum62_similarity(gen, ref))
                blosum_norm_scores.append(blosum62_normalized(gen, ref))

        if lev_distances:
            metrics["levenshtein_mean"] = float(np.mean(lev_distances))
            metrics["levenshtein_std"] = float(np.std(lev_distances))
            metrics["identity_mean"] = float(np.mean(identities))
            metrics["identity_std"] = float(np.std(identities))
            metrics["blosum62_mean"] = float(np.mean(blosum_scores))
            metrics["blosum62_std"] = float(np.std(blosum_scores))
            metrics["blosum62_normalized"] = float(np.mean(blosum_norm_scores))

        # ===== Position-Specific Accuracy =====
        if self.position_analyzer is not None:
            self.position_analyzer.reset()
            for gen, ref in zip(gen_clean, ref_clean):
                if gen and ref:
                    self.position_analyzer.update(gen, ref)

            metrics["mean_position_accuracy"] = self.position_analyzer.get_mean_position_accuracy()

            cdr3_acc = self.position_analyzer.get_cdr3_conserved_accuracy()
            metrics["cdr3_first3_accuracy"] = cdr3_acc["first3"]
            metrics["cdr3_last2_accuracy"] = cdr3_acc["last2"]

            # Top substitution errors
            top_errors = self.position_analyzer.get_top_substitution_errors(top_k=5)
            metrics["top_substitution_errors"] = top_errors

        # ===== Amino Acid Composition Metrics =====
        hydro_corrs = []
        charge_errors = []
        js_divergences = []

        for gen, ref in zip(gen_clean, ref_clean):
            if gen and ref and len(gen) >= 3 and len(ref) >= 3:
                # Hydrophobicity correlation
                hydro_corrs.append(hydrophobicity_correlation(gen, ref))

                # Charge error
                gen_charge = compute_net_charge(gen)
                ref_charge = compute_net_charge(ref)
                charge_errors.append(abs(gen_charge - ref_charge))

                # AA distribution divergence
                gen_dist = aa_frequency_distribution(gen)
                ref_dist = aa_frequency_distribution(ref)
                js_divergences.append(jensen_shannon_divergence(gen_dist, ref_dist))

        if hydro_corrs:
            metrics["hydrophobicity_corr_mean"] = float(np.mean(hydro_corrs))
            metrics["hydrophobicity_corr_std"] = float(np.std(hydro_corrs))
            metrics["charge_mae"] = float(np.mean(charge_errors))
            metrics["aa_distribution_js"] = float(np.mean(js_divergences))

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        generate_samples: bool = False,
        num_generate: int = 100,
        compute_detailed_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on generation task with comprehensive metrics.

        Args:
            model: TCRSeq2SeqModel
            dataloader: DataLoader for evaluation
            device: Device to run on
            generate_samples: Whether to generate samples for quality metrics
            num_generate: Number of samples to generate (default 100 for good statistics)
            compute_detailed_metrics: Whether to compute advanced biological metrics

        Returns:
            dict with all metrics
        """
        model.eval()

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        # For generation metrics
        generated_seqs = []
        reference_seqs = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            encoder_ids = batch["encoder_input_ids"].to(device)
            encoder_mask = batch["encoder_attention_mask"].to(device)
            decoder_ids = batch["decoder_input_ids"].to(device)
            decoder_mask = batch["decoder_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    encoder_input_ids=encoder_ids,
                    encoder_attention_mask=encoder_mask,
                    decoder_input_ids=decoder_ids,
                    decoder_attention_mask=decoder_mask,
                    labels=labels,
                )

            loss = outputs["loss"]

            # Count non-padding tokens
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            num_batches += 1

            # Generate samples for first num_generate batches
            if generate_samples and len(generated_seqs) < num_generate:
                # Generate for entire batch to be more efficient
                batch_size = encoder_ids.size(0)
                samples_needed = min(batch_size, num_generate - len(generated_seqs))

                gen_output = greedy_decode(
                    model.module if hasattr(model, "module") else model,
                    encoder_ids[:samples_needed],
                    encoder_mask[:samples_needed],
                    bos_id=self.tokenizer.cls_token_id,
                    eos_id=self.tokenizer.eos_token_id,
                    max_length=decoder_ids.size(1),
                )

                # Decode sequences
                for i in range(samples_needed):
                    gen_seq = self.tokenizer.decode(
                        gen_output[i].tolist(),
                        skip_special_tokens=True,
                    )
                    ref_seq = self.tokenizer.decode(
                        labels[i][labels[i] != -100].tolist(),
                        skip_special_tokens=True,
                    )

                    generated_seqs.append(gen_seq)
                    reference_seqs.append(ref_seq)

        # Calculate base metrics
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow

        metrics = {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
            "num_batches": num_batches,
            "total_tokens": total_tokens,
        }

        # Calculate generation metrics if samples were generated
        if generated_seqs:
            # Clean sequences for comparison
            gen_clean = [s.replace(" ", "") for s in generated_seqs]
            ref_clean = [s.replace(" ", "") for s in reference_seqs]

            exact_matches = sum(
                1 for g, r in zip(gen_clean, ref_clean)
                if g == r
            )
            metrics["exact_match_rate"] = exact_matches / len(generated_seqs)
            metrics["num_generated"] = len(generated_seqs)

            # Length statistics
            gen_lengths = [len(s) for s in gen_clean]
            ref_lengths = [len(s) for s in ref_clean]

            metrics["avg_gen_length"] = float(np.mean(gen_lengths))
            metrics["avg_ref_length"] = float(np.mean(ref_lengths))
            metrics["length_ratio"] = float(np.mean(gen_lengths)) / max(float(np.mean(ref_lengths)), 1)

            # Compute detailed biological metrics
            if compute_detailed_metrics and self.compute_detailed_metrics:
                detailed_metrics = self._compute_sequence_metrics(
                    generated_seqs, reference_seqs
                )
                metrics.update(detailed_metrics)

            # Sample outputs for logging (first 3)
            metrics["sample_generated"] = generated_seqs[:3]
            metrics["sample_reference"] = reference_seqs[:3]

        return metrics


# =============================================================================
# Trainer: TCRSeq2SeqTrainer
# =============================================================================


class TCRSeq2SeqTrainer(BaseTCRTrainer):
    """
    Trainer for TCR conditional sequence generation.

    Extends BaseTCRTrainer with seq2seq-specific functionality:
    - Encoder-decoder architecture with ESM2 encoder
    - Cross-attention based generation
    - Comprehensive biological metrics for evaluation
    - Support for CUDA and Trainium backends

    Features:
    - DDP/XLA support for multi-device training
    - Mixed precision (bfloat16)
    - LoRA for encoder (optional)
    - Gradient checkpointing
    - Warmup + cosine LR schedule
    - Early stopping
    - Checkpointing with best model tracking
    - Overfit check mode
    """

    def __init__(self, config: Dict[str, Any], backend: Optional[AcceleratorBackend] = None):
        """
        Initialize seq2seq trainer.

        Args:
            config: Training configuration
            backend: Accelerator backend (auto-detected if None)
        """
        # Initialize base trainer (handles device setup, logging, etc.)
        super().__init__(config, backend)

        # Seq2seq specific: parse task
        task_str = self.config.get("task", "ALPHA")
        self.task = GenerationTask[task_str.upper()]

        # Evaluator will be set up during data setup
        self.evaluator = None

        # BLOSUM loss function (initialized after tokenizer is set up)
        self.blosum_loss_fn = None

        # CDR annotation cache
        self.cdr_cache = CDRAnnotationCache()

    def _create_model(self) -> nn.Module:
        """Create TCRSeq2SeqModel with backend-appropriate settings."""
        model_name = self.config.get("model_name", "facebook/esm2_t33_650M_UR50D")

        self._log(f"Loading tokenizer and model: {model_name}")

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get backend-specific model loading kwargs
        load_kwargs = self._get_model_load_kwargs()

        # Create model with backend-appropriate attention implementation
        model = TCRSeq2SeqModel(
            encoder_model_name=model_name,
            decoder_layers=self.config.get("decoder_layers", 6),
            decoder_heads=self.config.get("decoder_heads", 20),
            decoder_dim=self.config.get("decoder_dim", 1280),
            decoder_ffn_dim=self.config.get("decoder_ffn_dim", 5120),
            dropout=self.config.get("dropout", 0.1),
            decoder_warm_start=self.config.get("decoder_warm_start", False),
            self_attn_drop_initial=self.config.get("self_attn_drop_initial", 0.0),
            self_attn_drop_final=self.config.get("self_attn_drop_final", 0.0),
            self_attn_drop_anneal_epochs=self.config.get("self_attn_drop_anneal_epochs", 3),
            **load_kwargs,
        )

        # Apply LoRA if requested
        if self.config.get("use_lora", False):
            if not PEFT_AVAILABLE:
                self._log("Warning: peft not available, skipping LoRA")
            else:
                model = apply_lora_to_encoder(model, self.config)
                self._log("Applied LoRA to encoder")
                if self._is_main_process():
                    model.encoder.print_trainable_parameters()

        # Load pre-trained encoder checkpoint if provided (for SFT on custom models)
        encoder_checkpoint = self.config.get("encoder_checkpoint")
        if encoder_checkpoint:
            self._log(f"Loading encoder checkpoint: {encoder_checkpoint}")
            model = load_encoder_checkpoint(
                model,
                encoder_checkpoint,
                strict=False,
                verbose=self._is_main_process(),
            )
            self._log("Encoder checkpoint loaded for SFT")

            # Merge foundation LoRA into base weights, then apply fresh LoRA for SFT.
            # This "bakes in" the foundation's learned representations and gives SFT
            # a fresh set of LoRA adapters to learn the generation task.
            if self.config.get("use_lora", False) and PEFT_AVAILABLE:
                self._log("Merging foundation LoRA into base weights...")
                model.encoder = model.encoder.merge_and_unload()
                # Remove residual PEFT metadata so re-application doesn't warn
                if hasattr(model.encoder, "peft_config"):
                    delattr(model.encoder, "peft_config")
                self._log("Applying fresh LoRA for SFT")
                model = apply_lora_to_encoder(model, self.config)
                if self._is_main_process():
                    model.encoder.print_trainable_parameters()

        # Freeze encoder if not using LoRA
        if not self.config.get("use_lora", False) and self.config.get("freeze_encoder", True):
            for param in model.encoder.parameters():
                param.requires_grad = False
            self._log("Encoder frozen (no LoRA)")

        # Enable gradient checkpointing for encoder (not supported on XLA/Trainium)
        if self.config.get("gradient_checkpointing", True):
            if self.backend.name == "xla":
                self._log("Gradient checkpointing disabled (not supported on XLA/Trainium)")
            else:
                model.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                self._log("Gradient checkpointing enabled")

        return model

    def _create_datasets(self) -> Tuple[Dataset, Dataset]:
        """Create train and validation datasets."""
        # Get permutation keys from config (can be None for auto-discovery)
        permutation_keys = self.config.get("permutation_keys")
        if isinstance(permutation_keys, str):
            permutation_keys = [permutation_keys]
        # Handle empty list from nargs="*" as None for auto-discovery
        if permutation_keys is not None and len(permutation_keys) == 0:
            permutation_keys = None

        train_dataset = TCRSeq2SeqDataset(
            data_path=self.config["data_path"],
            permutation_keys=permutation_keys,
            task=self.task,
            split="train",
            local_rank=self.local_rank,
        )

        val_dataset = TCRSeq2SeqDataset(
            data_path=self.config["data_path"],
            permutation_keys=permutation_keys,
            task=self.task,
            split="val",
            local_rank=self.local_rank,
        )

        return train_dataset, val_dataset

    def _create_collator(self) -> Seq2SeqCollator:
        """Create seq2seq data collator."""
        # Ensure tokenizer is set up
        if self.tokenizer is None:
            model_name = self.config.get("model_name", "facebook/esm2_t33_650M_UR50D")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        return Seq2SeqCollator(
            tokenizer=self.tokenizer,
            max_encoder_length=self.config.get("max_encoder_length", 1024),
            max_decoder_length=self.config.get("max_decoder_length", 350),
        )

    def _compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute biologically-informed loss for seq2seq training.

        Loss components (all configurable):
        1. CDR-weighted CE: Weight CDR3 > CDR1/2 > Framework via ANARCI positions
        2. BLOSUM62 soft loss: Penalize biologically dissimilar substitutions more
        3. Auxiliary losses: Identity, length consistency

        For PEPTIDE task, CDR weighting is disabled (peptides don't have CDR regions).
        """
        outputs = model(
            encoder_input_ids=batch["encoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )

        logits = outputs["logits"]  # (batch, seq_len, vocab_size)
        labels = batch["labels"]     # (batch, seq_len)
        batch_size, seq_len, vocab_size = logits.shape

        # =====================================================
        # COMPONENT 1: CDR-Weighted Cross-Entropy Loss
        # Only for TCR generation (ALPHA/BETA), not PEPTIDE
        # =====================================================
        use_cdr_weighting = (
            self.config.get("use_cdr_weighting", False) and
            self.task != GenerationTask.PEPTIDE  # Peptides don't have CDR regions
        )

        if use_cdr_weighting:
            # Get CDR annotations from batch (if available from collator)
            # If not in batch, compute on-the-fly from labels using ANARCI (cached)
            cdr_annotations = batch.get("cdr_annotations")
            if cdr_annotations is None and ANARCI_AVAILABLE:
                cdr_annotations = self._get_cdr_annotations_from_labels(labels)

            # Ensure cdr_annotations is a list (default to all None if not available)
            if cdr_annotations is None:
                cdr_annotations = [None] * batch_size

            # Build per-position weights for each sequence in batch
            position_weights = torch.ones(batch_size, seq_len, device=self.device)

            for b in range(batch_size):
                annot = cdr_annotations[b] if b < len(cdr_annotations) else None
                if annot is not None:
                    weights_b = build_cdr_position_weights(
                        seq_len=seq_len,
                        cdr_annotations=annot,
                        cdr1_weight=self.config.get("cdr1_weight", 2.0),
                        cdr2_weight=self.config.get("cdr2_weight", 2.0),
                        cdr3_weight=self.config.get("cdr3_weight", 4.0),
                        framework_weight=self.config.get("framework_weight", 1.0),
                        device=self.device,
                    )
                    position_weights[b] = weights_b

            # Weighted cross-entropy per position
            ce_per_position = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction='none'
            ).view(batch_size, seq_len)

            valid_mask = (labels != -100).float()
            weighted_sum = (ce_per_position * position_weights * valid_mask).sum()
            weight_sum = (position_weights * valid_mask).sum().clamp(min=1)
            weighted_ce = weighted_sum / weight_sum
        else:
            # Standard CE loss (used for PEPTIDE task or when CDR weighting disabled)
            weighted_ce = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        # =====================================================
        # COMPONENT 2: BLOSUM62 Soft Loss
        # Penalize biologically dissimilar substitutions more
        # =====================================================
        blosum_loss = torch.tensor(0.0, device=self.device)
        blosum_alpha = self.config.get("blosum_alpha", 0.0)

        if self.config.get("use_blosum_loss", False) and blosum_alpha > 0:
            if hasattr(self, 'blosum_loss_fn') and self.blosum_loss_fn is not None:
                blosum_loss = self.blosum_loss_fn(logits, labels)

        # =====================================================
        # COMPONENT 3: Auxiliary Losses (identity, length)
        # CRITICAL: Must be differentiable - no argmax!
        # =====================================================
        aux_loss = torch.tensor(0.0, device=self.device)

        if self.config.get("use_auxiliary_losses", False):
            # Differentiable soft identity loss using softmax probabilities
            # (argmax breaks gradients - we use probability of correct label instead)
            probs = F.softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)

            # Handle padding: replace -100 with 0 for gather, then mask later
            valid_mask = (labels != -100).float()
            n_valid = valid_mask.sum().clamp(min=1)
            labels_clamped = labels.clamp(min=0)  # Replace -100 with 0 for indexing

            # Get probability assigned to correct label at each position
            target_probs = probs.gather(2, labels_clamped.unsqueeze(2)).squeeze(2)  # (batch, seq_len)

            # Soft identity: mean probability of correct label (masked)
            soft_identity = (target_probs * valid_mask).sum() / n_valid
            identity_loss = 1.0 - soft_identity  # Minimize this = maximize target prob

            # Differentiable length consistency loss using expected EOS position
            # Use softmax to compute expected EOS position (soft argmax)
            eos_id = self.tokenizer.eos_token_id if self.tokenizer else 2
            eos_probs = probs[:, :, eos_id]  # (batch, seq_len) - prob of EOS at each position

            # Position indices
            positions = torch.arange(seq_len, device=self.device, dtype=torch.float).unsqueeze(0)  # (1, seq_len)

            # Expected EOS position = sum(position * EOS_prob) / sum(EOS_prob)
            # Add small epsilon to avoid division by zero
            eos_prob_sum = eos_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
            pred_expected_eos = (positions * eos_probs).sum(dim=1) / eos_prob_sum.squeeze(1)

            # Label EOS position (non-differentiable, used as target)
            label_eos_pos = torch.where(
                labels == eos_id,
                torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1),
                torch.full((batch_size, seq_len), seq_len, device=self.device)
            ).min(dim=1).values.float()

            length_loss = F.l1_loss(pred_expected_eos, label_eos_pos)

            aux_loss = (
                self.config.get("aux_identity_weight", 0.05) * identity_loss +
                self.config.get("aux_length_weight", 0.02) * length_loss
            )

        # =====================================================
        # COMBINE ALL LOSSES
        # =====================================================
        # Main loss: (1 - blosum_alpha) * CDR_weighted_CE + blosum_alpha * BLOSUM_loss
        if blosum_alpha > 0:
            main_loss = (1.0 - blosum_alpha) * weighted_ce + blosum_alpha * blosum_loss
        else:
            main_loss = weighted_ce

        # Add auxiliary losses
        total_loss = main_loss + aux_loss

        return {
            "loss": total_loss,
            "ce_loss": weighted_ce.detach(),
            "blosum_loss": blosum_loss.detach() if isinstance(blosum_loss, torch.Tensor) else torch.tensor(0.0),
            "aux_loss": aux_loss.detach() if isinstance(aux_loss, torch.Tensor) else torch.tensor(0.0),
        }

    def _compute_stopping_metric(self, val_metrics: Dict[str, float]) -> float:
        """
        Compute composite early stopping metric from biological metrics.

        Lower is better. Combines:
        - Loss (lower is better)
        - Identity (higher is better -> invert)
        - Position accuracy (higher is better -> invert)
        - Hydrophobicity correlation (higher is better -> invert)

        Returns:
            Composite metric (lower is better)
        """
        # Get weights from config
        loss_weight = self.config.get("stopping_loss_weight", 0.4)
        identity_weight = self.config.get("stopping_identity_weight", 0.3)
        position_weight = self.config.get("stopping_position_weight", 0.2)
        hydro_weight = self.config.get("stopping_hydro_weight", 0.1)

        # Normalize weights
        total_weight = loss_weight + identity_weight + position_weight + hydro_weight
        loss_weight /= total_weight
        identity_weight /= total_weight
        position_weight /= total_weight
        hydro_weight /= total_weight

        # Get metrics (with defaults)
        loss = val_metrics.get("eval_loss", 1.0)
        identity = val_metrics.get("identity_mean", 0.0)
        position_acc = val_metrics.get("mean_position_accuracy", 0.0)
        hydro_corr = val_metrics.get("hydrophobicity_corr_mean", 0.0)

        # Compute composite (lower is better)
        # For metrics where higher is better, use (1 - metric)
        composite = (
            loss_weight * loss +
            identity_weight * (1.0 - identity) +
            position_weight * (1.0 - position_acc) +
            hydro_weight * (1.0 - (hydro_corr + 1.0) / 2.0)  # Normalize hydro_corr from [-1,1] to [0,1]
        )

        return composite

    def _get_cdr_annotations_from_labels(
        self,
        labels: torch.Tensor
    ) -> List[Optional[Dict[str, Tuple[int, int]]]]:
        """
        Compute CDR annotations from label tokens using ANARCI.

        Decodes labels to sequences and uses cached ANARCI annotation.

        Args:
            labels: (batch, seq_len) label tensor

        Returns:
            List of CDR annotation dicts, one per batch item
        """
        if not ANARCI_AVAILABLE or self.tokenizer is None:
            return [None] * labels.shape[0]

        annotations = []
        for b in range(labels.shape[0]):
            # Get valid tokens (not -100 padding)
            valid_tokens = labels[b][labels[b] != -100].tolist()
            if len(valid_tokens) < 10:  # Too short for meaningful annotation
                annotations.append(None)
                continue

            # Decode to sequence
            try:
                sequence = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                sequence = sequence.replace(" ", "")  # Remove spaces

                if len(sequence) < 50:  # Too short for full TCR
                    annotations.append(None)
                    continue

                # Determine chain type from task
                chain_type = "A" if self.task == GenerationTask.ALPHA else "B"

                # Get CDR positions (cached)
                cdr_pos = self.cdr_cache.get_or_compute(sequence, chain_type)
                annotations.append(cdr_pos)

            except Exception:
                annotations.append(None)

        return annotations

    def setup_data(self, include_val: bool = True) -> None:
        """
        Setup data with seq2seq-specific evaluator and biological loss functions.

        Overrides base class to add:
        - GenerationEvaluator for comprehensive metrics
        - BLOSUM62SoftLoss for biological similarity-aware training
        """
        # Call base class setup
        super().setup_data(include_val)

        # Setup seq2seq-specific evaluator
        if include_val:
            self.evaluator = GenerationEvaluator(
                self.tokenizer,
                compute_detailed_metrics=self.config.get("compute_detailed_metrics", True),
            )

        # Setup BLOSUM loss function if enabled
        if self.config.get("use_blosum_loss", False) and self.tokenizer is not None:
            self.blosum_loss_fn = BLOSUM62SoftLoss(
                tokenizer=self.tokenizer,
                temperature=self.config.get("blosum_temperature", 1.0),
                alpha=self.config.get("blosum_alpha", 0.1),
            )
            # Move to device
            if hasattr(self, 'device'):
                self.blosum_loss_fn = self.blosum_loss_fn.to(self.device)
            self._log(f"Initialized BLOSUM62SoftLoss (alpha={self.config.get('blosum_alpha', 0.1)})")

        # Log CDR weighting configuration
        if self.config.get("use_cdr_weighting", False):
            if self.task == GenerationTask.PEPTIDE:
                self._log("Note: CDR weighting disabled for PEPTIDE task (peptides have no CDR regions)")
            else:
                self._log(f"CDR weighting enabled: CDR3={self.config.get('cdr3_weight', 4.0)}, "
                         f"CDR1/2={self.config.get('cdr1_weight', 2.0)}/{self.config.get('cdr2_weight', 2.0)}, "
                         f"Framework={self.config.get('framework_weight', 1.0)}")
                if not ANARCI_AVAILABLE:
                    self._log("Warning: ANARCI not available - CDR positions will use fallback (uniform weights)")

    # Keep legacy method name for backward compatibility
    def _setup_data(self, include_val: bool = True):
        """Legacy method name - calls setup_data."""
        self.setup_data(include_val)

    def _setup_optimizer(self, lr: Optional[float] = None):
        """Legacy method - calls setup_optimizer."""
        self.setup_optimizer(lr)

    def _setup_scheduler(self, num_training_steps: int):
        """Legacy method - calls setup_scheduler."""
        self.setup_scheduler(num_training_steps)

    def overfit_single_batch(self):
        """
        Overfit check: Train on a single batch to verify the pipeline works.

        This should drive loss to near zero and demonstrate the model can learn.
        """
        # Setup model and data if not already done
        if self.model is None:
            self.setup_model()
        if self.train_loader is None:
            self.setup_data(include_val=False)

        self._log("\n" + "=" * 60)
        self._log("SINGLE BATCH OVERFIT CHECK")
        self._log("=" * 60)
        self._log("Expected: Loss should decrease significantly")
        self._log("=" * 60 + "\n")

        num_steps = self.config.get("overfit_steps", 500)
        lr = self.config.get("overfit_lr") or self.config.get("learning_rate") or 1e-4

        self._setup_optimizer(lr=lr)
        self.model.train()

        # Get a single batch
        batch = next(iter(self.train_loader))
        encoder_ids = batch["encoder_input_ids"].to(self.device)
        encoder_mask = batch["encoder_attention_mask"].to(self.device)
        decoder_ids = batch["decoder_input_ids"].to(self.device)
        decoder_mask = batch["decoder_attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        self._log(f"[Batch Info]")
        self._log(f"  Encoder shape: {encoder_ids.shape}")
        self._log(f"  Decoder shape: {decoder_ids.shape}")
        self._log(f"  Learning rate: {lr}")
        self._log("")

        initial_loss = None

        for step in range(num_steps):
            self.optimizer.zero_grad()

            with self.backend.autocast_context(self.backend.get_model_dtype()):
                outputs = self.model(
                    encoder_input_ids=encoder_ids,
                    encoder_attention_mask=encoder_mask,
                    decoder_input_ids=decoder_ids,
                    decoder_attention_mask=decoder_mask,
                    labels=labels,
                )

            loss = outputs["loss"]
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get("max_grad_norm", 1.0)
            )

            self.backend.optimizer_step(self.optimizer, self.model)

            if step == 0:
                initial_loss = loss.item()

            if step % 10 == 0 or step == num_steps - 1:
                self._log(
                    f"Step {step:4d}: loss={loss.item():.4f}, "
                    f"ppl={math.exp(min(loss.item(), 100)):.2f}, "
                    f"grad_norm={grad_norm:.4f}"
                )

        final_loss = loss.item()

        self._log("\n" + "=" * 60)
        self._log("OVERFIT CHECK RESULTS")
        self._log("=" * 60)
        self._log(f"Initial loss: {initial_loss:.4f}")
        self._log(f"Final loss:   {final_loss:.4f}")
        self._log(f"Loss reduction: {100 * (initial_loss - final_loss) / initial_loss:.1f}%")
        self._log(f"Initial perplexity: {math.exp(min(initial_loss, 100)):.2f}")
        self._log(f"Final perplexity:   {math.exp(min(final_loss, 100)):.2f}")

        if final_loss < initial_loss * 0.5:
            self._log("\nSUCCESS: Model can overfit a single batch!")
            self._log("The encoder-decoder architecture is working correctly.")
        else:
            self._log("\nWARNING: Loss reduction is limited.")
            self._log("Consider checking: gradients, learning rate, architecture")

        self._log("=" * 60 + "\n")

    def _save_checkpoint(self, epoch: int, step: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        if not self._is_main_process():
            return

        model_to_save = self.model.module if self.is_distributed else self.model

        checkpoint = {
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
            "metrics": metrics,
            "config": self.config,
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self._log(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self._log(f"New best model saved! Loss: {metrics.get('eval_loss', 'N/A'):.4f}")

        # Cleanup old checkpoints (keep 3)
        checkpoints = sorted(self.output_dir.glob("checkpoint_step_*.pt"))
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()

    def train(self):
        """Full training loop with validation, checkpointing, and early stopping."""
        # Setup model and data if not already done
        if self.model is None:
            self.setup_model()
        if self.train_loader is None:
            self.setup_data(include_val=True)

        self._log("\n" + "=" * 60)
        self._log("STARTING FULL TRAINING")
        self._log("=" * 60 + "\n")

        num_epochs = self.config.get("num_epochs", 10)
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        eval_steps = self.config.get("eval_steps", 1000)
        save_steps = self.config.get("save_steps", 5000)
        log_steps = self.config.get("log_steps", 100)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        patience = self.config.get("patience", 5)

        num_training_steps = len(self.train_loader) * num_epochs // grad_accum_steps
        self._setup_optimizer()
        self._setup_scheduler(num_training_steps)

        self._log(f"Training configuration:")
        self._log(f"  Epochs: {num_epochs}")
        self._log(f"  Batch size: {self.config.get('batch_size', 16)}")
        self._log(f"  Gradient accumulation: {grad_accum_steps}")
        self._log(f"  Effective batch size: {self.config.get('batch_size', 16) * grad_accum_steps * self.world_size}")
        self._log(f"  Total training steps: {num_training_steps}")
        self._log(f"  Early stopping patience: {patience}")
        self._log("")

        patience_counter = 0
        running_loss = 0.0
        running_batches = 0

        # Setup wandb
        use_wandb = self.config.get("report_to") == "wandb" and self._is_main_process() and WANDB_AVAILABLE
        if use_wandb:
            wandb.init(
                project=self.config.get("wandb_project", "tcr-seq2seq"),
                name=self.config.get("wandb_run_name"),
                config=self.config,
                settings=wandb.Settings(init_timeout=300),  # 5 min timeout for slow connections
            )

        self.model.train()

        for epoch in range(num_epochs):
            if self.is_distributed:
                self.train_loader.sampler.set_epoch(epoch)

            self._log(f"\n{'='*60}")
            self._log(f"Epoch {epoch + 1}/{num_epochs}")
            self._log(f"{'='*60}")

            # Update self-attention dropout schedule for cross-attention forcing
            model_to_update = self.model.module if self.is_distributed else self.model
            model_to_update.set_epoch(epoch)

            epoch_loss = 0.0
            epoch_steps = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}",
                disable=not self._is_main_process(),
            )

            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(progress_bar):
                encoder_ids = batch["encoder_input_ids"].to(self.device)
                encoder_mask = batch["encoder_attention_mask"].to(self.device)
                decoder_ids = batch["decoder_input_ids"].to(self.device)
                decoder_mask = batch["decoder_attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with self.backend.autocast_context(self.backend.get_model_dtype()):
                    outputs = self.model(
                        encoder_input_ids=encoder_ids,
                        encoder_attention_mask=encoder_mask,
                        decoder_input_ids=decoder_ids,
                        decoder_attention_mask=decoder_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"] / grad_accum_steps

                loss.backward()

                epoch_loss += outputs["loss"].item()
                epoch_steps += 1

                if (batch_idx + 1) % grad_accum_steps == 0:
                    running_loss += outputs["loss"].item()
                    running_batches += 1

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=max_grad_norm
                    )

                    self.backend.optimizer_step(self.optimizer, self.model)
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % log_steps == 0 and self._is_main_process():
                        avg_loss = running_loss / running_batches
                        lr = self.scheduler.get_last_lr()[0]

                        self._log(
                            f"Step {self.global_step}: loss={avg_loss:.4f}, "
                            f"ppl={math.exp(min(avg_loss, 100)):.2f}, "
                            f"lr={lr:.2e}, grad_norm={grad_norm:.4f}"
                        )

                        if use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/perplexity": math.exp(min(avg_loss, 100)),
                                "train/learning_rate": lr,
                                "train/grad_norm": grad_norm,
                                "train/global_step": self.global_step,
                            })

                        running_loss = 0.0
                        running_batches = 0

                    # Evaluation
                    if self.global_step % eval_steps == 0:
                        self._log(f"\n--- Validation at step {self.global_step} ---")
                        val_metrics = self.evaluator.evaluate(
                            self.model,
                            self.val_loader,
                            self.device,
                            generate_samples=True,
                            num_generate=self.config.get("num_generate", 100),
                            compute_detailed_metrics=self.config.get("compute_detailed_metrics", True),
                        )

                        # Log base metrics
                        self._log(
                            f"Val: loss={val_metrics['eval_loss']:.4f}, "
                            f"ppl={val_metrics['perplexity']:.2f}"
                        )

                        if "exact_match_rate" in val_metrics:
                            self._log(f"     exact_match={val_metrics['exact_match_rate']:.4f}")

                        # Log sequence similarity metrics
                        if "identity_mean" in val_metrics:
                            self._log(
                                f"     identity={val_metrics['identity_mean']:.4f} (+/-{val_metrics.get('identity_std', 0):.3f}), "
                                f"levenshtein={val_metrics.get('levenshtein_mean', 0):.2f}"
                            )

                        # Log position-specific metrics
                        if "mean_position_accuracy" in val_metrics:
                            self._log(
                                f"     pos_acc={val_metrics['mean_position_accuracy']:.4f}, "
                                f"cdr3_first3={val_metrics.get('cdr3_first3_accuracy', 0):.4f}, "
                                f"cdr3_last2={val_metrics.get('cdr3_last2_accuracy', 0):.4f}"
                            )

                        # Log composition metrics
                        if "hydrophobicity_corr_mean" in val_metrics:
                            self._log(
                                f"     hydro_corr={val_metrics['hydrophobicity_corr_mean']:.4f}, "
                                f"charge_mae={val_metrics.get('charge_mae', 0):.3f}, "
                                f"aa_js_div={val_metrics.get('aa_distribution_js', 0):.4f}"
                            )

                        if "sample_generated" in val_metrics:
                            self._log("\n[Sample outputs]")
                            for i, (gen, ref) in enumerate(zip(
                                val_metrics["sample_generated"][:2],
                                val_metrics["sample_reference"][:2]
                            )):
                                self._log(f"  [{i}] Gen: {gen[:80]}...")
                                self._log(f"      Ref: {ref[:80]}...")

                        if use_wandb:
                            wandb.log({
                                f"eval/{k}": v
                                for k, v in val_metrics.items()
                                if isinstance(v, (int, float))
                            } | {"eval/global_step": self.global_step})

                        # Check for improvement using biological metrics if enabled
                        if self.config.get("use_biological_stopping", False):
                            # Compute composite stopping metric (lower is better)
                            stopping_metric = self._compute_stopping_metric(val_metrics)
                            is_best = stopping_metric < getattr(self, 'best_stopping_metric', float('inf'))
                            if is_best:
                                self.best_stopping_metric = stopping_metric
                                self.best_val_loss = val_metrics["eval_loss"]
                                patience_counter = 0
                                self._log(f"New best composite metric: {stopping_metric:.4f}")
                            else:
                                patience_counter += 1
                                self._log(f"No improvement (composite={stopping_metric:.4f}). Patience: {patience_counter}/{patience}")
                        else:
                            # Standard: only check loss
                            is_best = val_metrics["eval_loss"] < self.best_val_loss
                            if is_best:
                                self.best_val_loss = val_metrics["eval_loss"]
                                patience_counter = 0
                            else:
                                patience_counter += 1
                                self._log(f"No improvement. Patience: {patience_counter}/{patience}")

                        self._save_checkpoint(epoch, self.global_step, val_metrics, is_best=is_best)

                        if patience_counter >= patience:
                            self._log(f"\nEarly stopping triggered at step {self.global_step}")
                            break

                        self.model.train()

                    # Periodic checkpoint
                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(
                            epoch, self.global_step,
                            {"train_loss": epoch_loss / epoch_steps}
                        )

                progress_bar.set_postfix({
                    "loss": f"{epoch_loss/max(epoch_steps, 1):.4f}",
                    "step": f"{self.global_step}",
                })

            if patience_counter >= patience:
                break

            # End of epoch logging
            self._log(f"\n--- End of Epoch {epoch + 1} ---")
            self._log(f"Train: loss={epoch_loss/epoch_steps:.4f}, ppl={math.exp(min(epoch_loss/epoch_steps, 100)):.2f}")

        # Final summary
        self._log("\n" + "=" * 60)
        self._log("TRAINING COMPLETE")
        self._log("=" * 60)
        self._log(f"Best validation loss: {self.best_val_loss:.4f}")
        self._log(f"Best validation perplexity: {math.exp(min(self.best_val_loss, 100)):.2f}")
        self._log(f"Total steps: {self.global_step}")
        self._log(f"Best model saved to: {self.output_dir / 'best_model.pt'}")

        if use_wandb:
            wandb.finish()

        # Save final model
        self._save_checkpoint(
            num_epochs - 1, self.global_step,
            {"final": True, "best_val_loss": self.best_val_loss}
        )


# =============================================================================
# CLI Arguments
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="TCR Conditional Sequence Generation Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Backend
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "cuda", "xla", "neuron", "trainium"],
                        help="Training backend: auto (detect), cuda (NVIDIA GPU), xla/neuron/trainium (AWS Trainium)")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to directory containing parquet files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--task", type=str, default="ALPHA",
                        choices=["ALPHA", "BETA", "PEPTIDE"],
                        help="Generation task")
    parser.add_argument("--permutation_keys", type=str, nargs="*",
                        default=None,
                        help="Permutation keys to filter sequences. "
                             "If not specified, auto-discovers keys matching task requirements.")
    parser.add_argument("--max_encoder_length", type=int, default=512,
                        help="Maximum encoder sequence length (512 for trn1.2xlarge, 1024 for larger instances)")
    parser.add_argument("--max_decoder_length", type=int, default=256,
                        help="Maximum decoder sequence length (256 for trn1.2xlarge, 350 for larger instances)")

    # Model (default to smallest ESM2 for trn1.2xlarge memory constraints)
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D",
                        help="Pretrained ESM2 model name (esm2_t6_8M for trn1.2xlarge, esm2_t33_650M for larger instances)")
    parser.add_argument("--decoder_layers", type=int, default=4,
                        help="Number of decoder transformer layers")
    parser.add_argument("--decoder_heads", type=int, default=4,
                        help="Number of decoder attention heads (4 for ESM2-8M, 20 for ESM2-650M)")
    parser.add_argument("--decoder_dim", type=int, default=320,
                        help="Decoder hidden dimension (320 for ESM2-8M, 1280 for ESM2-650M)")
    parser.add_argument("--decoder_ffn_dim", type=int, default=1280,
                        help="Decoder FFN intermediate dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--decoder_warm_start", action="store_true",
                        help="Initialize decoder self-attention and FFN from pre-trained encoder weights (fixes mode collapse)")

    # Cross-attention forcing (Phase 8 - fixes mode collapse when decoder ignores encoder)
    parser.add_argument("--self_attn_drop_initial", type=float, default=0.0,
                        help="Initial self-attention dropout probability (e.g., 0.5 to force cross-attention usage)")
    parser.add_argument("--self_attn_drop_final", type=float, default=0.0,
                        help="Final self-attention dropout probability after annealing (e.g., 0.1)")
    parser.add_argument("--self_attn_drop_anneal_epochs", type=int, default=3,
                        help="Number of epochs before annealing from initial to final dropout")

    # LoRA
    parser.add_argument("--use_lora", action="store_true",
                        help="Apply LoRA to encoder")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--freeze_encoder", action="store_true", default=True,
                        help="Freeze encoder if not using LoRA")

    # Custom encoder checkpoint (for SFT on pre-trained models)
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Path to pre-trained encoder checkpoint (.pt file). "
                             "Use this to fine-tune a custom pre-trained model (e.g., MLM-pretrained ESM2+LoRA). "
                             "The checkpoint should contain 'model_state_dict' with encoder weights.")

    # Training (defaults optimized for trn1.2xlarge with ESM2-8M)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per device (8 for trn1.2xlarge with ESM2-8M)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loader workers (2 for trn1.2xlarge to leave CPU for XLA compilation)")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (4 with batch_size=8 for effective batch=32)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing")

    # Logging and checkpointing
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")

    # Wandb
    parser.add_argument("--report_to", type=str, default="none",
                        choices=["wandb", "none"],
                        help="Reporting destination")
    parser.add_argument("--wandb_project", type=str, default="tcr-seq2seq",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")

    # Overfit check
    parser.add_argument("--overfit_check", action="store_true",
                        help="Run single-batch overfit check")
    parser.add_argument("--overfit_steps", type=int, default=500,
                        help="Number of steps for overfit check")
    parser.add_argument("--overfit_lr", type=float, default=None,
                        help="Learning rate for overfit check")

    # Enhanced evaluation
    parser.add_argument("--num_generate", type=int, default=100,
                        help="Number of samples to generate for evaluation metrics")
    parser.add_argument("--compute_detailed_metrics", action="store_true", default=True,
                        help="Compute advanced sequence metrics (BLOSUM62, position accuracy, etc.)")

    # ==========================================================================
    # Biological Loss Configuration
    # ==========================================================================

    # CDR Region Weighting (for ALPHA/BETA tasks only, not PEPTIDE)
    parser.add_argument("--use_cdr_weighting", action="store_true",
                        help="Enable CDR region-weighted loss (requires ANARCI). "
                             "Weights CDR3 > CDR1/2 > Framework regions.")
    parser.add_argument("--cdr_weight_preset", type=str, default=None,
                        choices=["uniform", "cdr3_focused", "all_cdr_equal", "extreme_cdr3"],
                        help="Preset CDR weight configuration. Overrides individual weights if set.")
    parser.add_argument("--cdr1_weight", type=float, default=2.0,
                        help="Loss weight for CDR1 region")
    parser.add_argument("--cdr2_weight", type=float, default=2.0,
                        help="Loss weight for CDR2 region")
    parser.add_argument("--cdr3_weight", type=float, default=4.0,
                        help="Loss weight for CDR3 region (most important for specificity)")
    parser.add_argument("--framework_weight", type=float, default=1.0,
                        help="Loss weight for framework regions (conserved)")

    # BLOSUM62 Soft Loss
    parser.add_argument("--use_blosum_loss", action="store_true",
                        help="Enable BLOSUM62 soft loss (penalize biologically dissimilar substitutions more)")
    parser.add_argument("--blosum_alpha", type=float, default=0.1,
                        help="BLOSUM loss mixing weight: (1-alpha)*CE + alpha*BLOSUM_KL")
    parser.add_argument("--blosum_temperature", type=float, default=1.0,
                        help="Temperature for BLOSUM score softmax normalization")

    # Auxiliary Losses
    parser.add_argument("--use_auxiliary_losses", action="store_true",
                        help="Enable auxiliary losses (identity, length consistency)")
    parser.add_argument("--aux_identity_weight", type=float, default=0.05,
                        help="Weight for soft sequence identity auxiliary loss")
    parser.add_argument("--aux_length_weight", type=float, default=0.02,
                        help="Weight for length consistency auxiliary loss")

    # Multi-objective Early Stopping
    parser.add_argument("--use_biological_stopping", action="store_true",
                        help="Use biological metrics for early stopping (not just loss)")
    parser.add_argument("--stopping_loss_weight", type=float, default=0.4,
                        help="Weight for loss in early stopping composite metric")
    parser.add_argument("--stopping_identity_weight", type=float, default=0.3,
                        help="Weight for identity in early stopping composite metric")
    parser.add_argument("--stopping_position_weight", type=float, default=0.2,
                        help="Weight for position accuracy in early stopping composite metric")
    parser.add_argument("--stopping_hydro_weight", type=float, default=0.1,
                        help="Weight for hydrophobicity correlation in early stopping")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    config = vars(args)

    # Apply CDR weight preset if specified
    if config.get("cdr_weight_preset"):
        preset_name = config["cdr_weight_preset"]
        if preset_name in CDR_WEIGHT_PRESETS:
            preset = CDR_WEIGHT_PRESETS[preset_name]
            config["cdr1_weight"] = preset["cdr1"]
            config["cdr2_weight"] = preset["cdr2"]
            config["cdr3_weight"] = preset["cdr3"]
            config["framework_weight"] = preset["framework"]
            print(f"Applied CDR weight preset '{preset_name}': {preset}")

    # Get backend from config (auto-detected if not specified)
    backend = get_backend(config.get("backend", "auto"))

    # Create trainer with backend
    trainer = TCRSeq2SeqTrainer(config, backend=backend)

    if args.overfit_check:
        # Use overfit check from base trainer
        trainer.overfit_single_batch()
    else:
        # Use full training loop
        trainer.train()

    # Cleanup distributed
    if trainer.is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
