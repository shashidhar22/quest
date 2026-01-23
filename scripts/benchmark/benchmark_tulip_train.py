#!/usr/bin/env python
"""
TULIP-TCR Benchmark Script - Train and Evaluate

This script trains a TULIP model from scratch on our data and evaluates
on test splits. TULIP uses a generative approach where binding affinity
is scored by the negative log-likelihood of generating sequences.

Reference: TULIP - A Transformer-based Unsupervised Language model for Interacting Peptides and T-cell receptors
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.benchmark.benchmark_metrics import compute_all_unified_metrics

# Add TULIP-TCR to path
TULIP_ROOT = "/home/sravisha/tcrbench_tools/TULIP-TCR"
sys.path.insert(0, TULIP_ROOT)

# Suppress warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel, BertConfig

# Task configurations for all 6 tasks
TULIP_TASK_CONFIGS = {
    "tra_peptide_mhc_one": {
        "mhc_class": "class_one",
        "mhc_cols": ["mhc_one_id"],
        "tcr_cols": ["tra"],  # trb will be <MIS>
        "description": "TCR alpha only + peptide + MHC-I -> binding",
    },
    "trb_peptide_mhc_one": {
        "mhc_class": "class_one",
        "mhc_cols": ["mhc_one_id"],
        "tcr_cols": ["trb"],  # tra will be <MIS>
        "description": "TCR beta only + peptide + MHC-I -> binding",
    },
    "tra_trb_peptide_mhc_one": {
        "mhc_class": "class_one",
        "mhc_cols": ["mhc_one_id"],
        "tcr_cols": ["tra", "trb"],
        "description": "TCR alpha+beta paired + peptide + MHC-I -> binding",
    },
    "tra_peptide_mhc_two": {
        "mhc_class": "class_two",
        "mhc_cols": ["mhc_one_id", "mhc_two_id"],
        "tcr_cols": ["tra"],  # trb will be <MIS>
        "description": "TCR alpha only + peptide + MHC-II -> binding",
    },
    "trb_peptide_mhc_two": {
        "mhc_class": "class_two",
        "mhc_cols": ["mhc_one_id", "mhc_two_id"],
        "tcr_cols": ["trb"],  # tra will be <MIS>
        "description": "TCR beta only + peptide + MHC-II -> binding",
    },
    "tra_trb_peptide_mhc_two": {
        "mhc_class": "class_two",
        "mhc_cols": ["mhc_one_id", "mhc_two_id"],
        "tcr_cols": ["tra", "trb"],
        "description": "TCR alpha+beta paired + peptide + MHC-II -> binding",
    },
}

TEST_SPLITS = [
    "test_seen_epitope",
    "test_unseen_tcr_seen_epitope",
    "test_unseen_epitope",
    "test_unseen_allele",
]


def format_mhc_for_tulip(mhc_id: str, mhc_class: str) -> str:
    """
    Convert our MHC format to TULIP format.

    Input: 'A*02:01' or 'DRA*01:01_DRB1*01:01'
    Output: 'HLA-A*02:01' or 'HLA-DRA*01:01/DRB1*01:01'
    """
    if pd.isna(mhc_id) or mhc_id == '':
        return '<MIS>'

    if mhc_class == "class_one":
        if not mhc_id.startswith('HLA-'):
            return f"HLA-{mhc_id}"
        return mhc_id
    else:
        parts = mhc_id.split('_')
        if len(parts) == 2:
            alpha, beta = parts
            if not alpha.startswith('HLA-'):
                alpha = f"HLA-{alpha}"
            return f"{alpha}/{beta}"
        return mhc_id


def convert_parquet_to_tulip_df(
    parquet_path: str,
    task_config: Dict,
    add_binder_column: bool = True,
    use_mhc: bool = True,
) -> pd.DataFrame:
    """
    Convert parquet file to TULIP DataFrame format.

    Args:
        parquet_path: Path to parquet file
        task_config: Task configuration dictionary
        add_binder_column: Whether to add binder=1 column for all samples
        use_mhc: Whether to include MHC information (if False, MHC set to <MIS>)

    Returns:
        DataFrame with columns: CDR3a, CDR3b, peptide, MHC, binder
    """
    df = pd.read_parquet(parquet_path)

    tulip_df = pd.DataFrame(index=df.index)

    def safe_string(val):
        """Convert value to string, replacing None/NaN/empty with <MIS>."""
        if val is None or pd.isna(val) or val == '':
            return '<MIS>'
        return str(val)

    # Map TCR columns - use <MIS> for missing chains
    if "tra" in task_config["tcr_cols"]:
        tulip_df["CDR3a"] = df["tra"].apply(safe_string)
    else:
        tulip_df["CDR3a"] = ['<MIS>'] * len(df)

    if "trb" in task_config["tcr_cols"]:
        tulip_df["CDR3b"] = df["trb"].apply(safe_string)
    else:
        tulip_df["CDR3b"] = ['<MIS>'] * len(df)

    # Peptide column
    tulip_df["peptide"] = df["peptide"].apply(safe_string)

    # MHC column - combine for MHC-II or set to <MIS> if not using MHC
    if use_mhc:
        mhc_class = task_config["mhc_class"]
        mhc_cols = task_config["mhc_cols"]

        if len(mhc_cols) == 1:
            mhc_values = df[mhc_cols[0]].fillna('')
        else:
            col1 = df[mhc_cols[0]].fillna('')
            col2 = df[mhc_cols[1]].fillna('')
            mhc_values = col1.astype(str) + "_" + col2.astype(str)

        tulip_df["MHC"] = mhc_values.apply(lambda x: format_mhc_for_tulip(x, mhc_class))
    else:
        # Set all MHC to <MIS> when not using MHC information
        tulip_df["MHC"] = '<MIS>'

    if add_binder_column:
        tulip_df["binder"] = 1

    tulip_df["original_peptide"] = df["peptide"]

    return tulip_df


def generate_negatives(positive_df: pd.DataFrame, neg_per_pos: int = 5) -> pd.DataFrame:
    """
    Generate negative samples using TULIP paper approach.

    For each positive TCR-peptide pair, generate N negatives by sampling TCRs
    from other peptides and pairing with the current peptide.
    """
    peptides = positive_df["peptide"].unique()

    peptide_alpha = {pep: set(positive_df[positive_df["peptide"] == pep]["CDR3a"].unique())
                     for pep in peptides}
    peptide_beta = {pep: set(positive_df[positive_df["peptide"] == pep]["CDR3b"].unique())
                    for pep in peptides}

    negatives = []

    for idx in range(len(positive_df)):
        row = positive_df.iloc[idx]
        target_peptide = row["peptide"]
        target_mhc = row["MHC"]

        neg_count = 0
        max_attempts = neg_per_pos * 10
        attempts = 0

        while neg_count < neg_per_pos and attempts < max_attempts:
            attempts += 1

            sample_idx = np.random.randint(len(positive_df))
            sample_row = positive_df.iloc[sample_idx]

            if sample_row["peptide"] == target_peptide:
                continue

            sample_alpha = sample_row["CDR3a"]
            sample_beta = sample_row["CDR3b"]

            is_alpha_binder = (sample_alpha != '<MIS>' and
                              sample_alpha in peptide_alpha.get(target_peptide, set()))
            is_beta_binder = (sample_beta != '<MIS>' and
                             sample_beta in peptide_beta.get(target_peptide, set()))

            if is_alpha_binder or is_beta_binder:
                continue

            negatives.append({
                "CDR3a": sample_alpha,
                "CDR3b": sample_beta,
                "peptide": target_peptide,
                "MHC": target_mhc,
                "binder": 0,
                "original_peptide": target_peptide,
            })
            neg_count += 1

    neg_df = pd.DataFrame(negatives)
    combined_df = pd.concat([positive_df, neg_df], ignore_index=True)

    return combined_df


def setup_tokenizer(tulip_root: str, use_mhc: bool = True):
    """
    Set up TULIP tokenizers.

    Args:
        tulip_root: Path to TULIP-TCR root directory
        use_mhc: Whether to use MHC tokenizer

    Returns:
        Tuple of (tokenizer, mhctok)
    """
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(tulip_root, "aatok/"))

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<MIS>'})
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '<CLS>'})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<EOS>'})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<MASK>'})

    from tokenizers.processors import TemplateProcessing
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<CLS> $A <EOS>",
        pair="<CLS> $A <MIS> $B:1 <EOS>:1",
        special_tokens=[
            ("<EOS>", 2),
            ("<CLS>", 3),
            ("<MIS>", 4),
        ],
    )

    if use_mhc:
        mhctok = AutoTokenizer.from_pretrained(os.path.join(tulip_root, "mhctok/"))
    else:
        mhctok = AutoTokenizer.from_pretrained(os.path.join(tulip_root, "nomhctok/"))

    return tokenizer, mhctok


def initialize_tulip_model(
    modelconfig: Dict,
    tokenizer,
    mhctok,
    device: torch.device,
):
    """
    Initialize a fresh TULIP model from config.

    Args:
        modelconfig: Model configuration dictionary
        tokenizer: Amino acid tokenizer
        mhctok: MHC tokenizer
        device: PyTorch device

    Returns:
        Initialized TULIP model
    """
    from src.multiTrans import TulipPetal, BertLastPooler, Tulip

    vocabsize = len(tokenizer._tokenizer.get_vocab())
    mhcvocabsize = len(mhctok._tokenizer.get_vocab())

    max_length = 50

    # Check if config has decoupled architecture
    if "num_attn_heads_encoder_cdr" in modelconfig:
        encoder_config_pep = BertConfig(
            vocab_size=vocabsize,
            max_position_embeddings=max_length,
            num_attention_heads=modelconfig["num_attn_heads_encoder_pep"],
            num_hidden_layers=modelconfig["num_hidden_layers_encoder_pep"],
            hidden_size=modelconfig["hidden_size"],
            type_vocab_size=1,
            pad_token_id=tokenizer.pad_token_id
        )

        encoder_config_cdr = BertConfig(
            vocab_size=vocabsize,
            max_position_embeddings=max_length,
            num_attention_heads=modelconfig["num_attn_heads_encoder_cdr"],
            num_hidden_layers=modelconfig["num_hidden_layers_encoder_cdr"],
            hidden_size=modelconfig["hidden_size"],
            type_vocab_size=1,
            pad_token_id=tokenizer.pad_token_id
        )

        decoder_config_pep = BertConfig(
            vocab_size=vocabsize,
            max_position_embeddings=max_length,
            num_attention_heads=modelconfig["num_attn_heads_decoder_pep"],
            num_hidden_layers=modelconfig["num_hidden_layers_decoder_pep"],
            hidden_size=modelconfig["hidden_size"],
            type_vocab_size=1,
            is_decoder=True,
            pad_token_id=tokenizer.pad_token_id
        )

        decoder_config_cdr = BertConfig(
            vocab_size=vocabsize,
            max_position_embeddings=max_length,
            num_attention_heads=modelconfig["num_attn_heads_decoder_cdr"],
            num_hidden_layers=modelconfig["num_hidden_layers_decoder_cdr"],
            hidden_size=modelconfig["hidden_size"],
            type_vocab_size=1,
            is_decoder=True,
            pad_token_id=tokenizer.pad_token_id
        )

        encoder_config_pep.mhc_vocab_size = mhcvocabsize
        encoder_config_cdr.mhc_vocab_size = mhcvocabsize

        decoder_config_cdr.add_cross_attention = True
        decoder_config_pep.add_cross_attention = True

        encoderA = BertModel(config=encoder_config_cdr)
        encoderB = BertModel(config=encoder_config_cdr)
        encoderE = BertModel(config=encoder_config_pep)

        decoderA = TulipPetal(config=decoder_config_cdr)
        decoderA.pooler = BertLastPooler(config=decoder_config_cdr)
        decoderB = TulipPetal(config=decoder_config_cdr)
        decoderB.pooler = BertLastPooler(config=decoder_config_cdr)
        decoderE = TulipPetal(config=decoder_config_pep)
        decoderE.pooler = BertLastPooler(config=decoder_config_pep)
    else:
        encoder_config = BertConfig(
            vocab_size=vocabsize,
            max_position_embeddings=max_length,
            num_attention_heads=modelconfig["num_attn_heads"],
            num_hidden_layers=modelconfig["num_hidden_layers"],
            hidden_size=modelconfig["hidden_size"],
            type_vocab_size=1,
            pad_token_id=tokenizer.pad_token_id
        )

        encoder_config.mhc_vocab_size = mhcvocabsize

        decoder_config = BertConfig(
            vocab_size=vocabsize,
            max_position_embeddings=max_length,
            num_attention_heads=modelconfig["num_attn_heads"],
            num_hidden_layers=modelconfig["num_hidden_layers"],
            hidden_size=modelconfig["hidden_size"],
            type_vocab_size=1,
            is_decoder=True,
            pad_token_id=tokenizer.pad_token_id
        )
        decoder_config.add_cross_attention = True

        encoderA = BertModel(config=encoder_config)
        encoderB = BertModel(config=encoder_config)
        encoderE = BertModel(config=encoder_config)

        decoderA = TulipPetal(config=decoder_config)
        decoderA.pooler = BertLastPooler(config=decoder_config)
        decoderB = TulipPetal(config=decoder_config)
        decoderB.pooler = BertLastPooler(config=decoder_config)
        decoderE = TulipPetal(config=decoder_config)
        decoderE.pooler = BertLastPooler(config=decoder_config)

    model = Tulip(
        encoderA=encoderA, encoderB=encoderB, encoderE=encoderE,
        decoderA=decoderA, decoderB=decoderB, decoderE=decoderE
    )
    model.skipMiss = True

    # Initialize weights with Xavier normal
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    model.to(device)

    return model


def train_one_epoch(
    model,
    optimizer,
    train_dataloader,
    criterion,
    masker,
    device: torch.device,
    masking_proba: float = 0.0,
):
    """
    Train for one epoch.

    Args:
        model: TULIP model
        optimizer: PyTorch optimizer
        train_dataloader: Training data loader
        criterion: Loss criterion
        masker: MLM masker
        device: PyTorch device
        masking_proba: Masking probability for training

    Returns:
        Tuple of losses
    """
    from src.multiTrans import compute_loss, MLM_Loss

    model.train()

    epoch_lm_lossA = 0
    epoch_lm_lossB = 0
    epoch_lm_lossE = 0
    epoch_mlm_lossA = 0
    epoch_mlm_lossB = 0
    epoch_mlm_lossE = 0
    count_A = 0
    count_B = 0
    count_E = 0

    for i, (peptide, alpha, beta, binder, mhc) in enumerate(train_dataloader):
        optimizer.zero_grad()

        peptide_input = peptide['input_ids']
        peptide_mask = peptide["attention_mask"]
        alpha_input = alpha['input_ids']
        alpha_mask = alpha["attention_mask"]
        beta_input = beta['input_ids']
        beta_mask = beta["attention_mask"]

        # Check which sequences are observed (not missing)
        alpha_observed_mask = alpha_input.clone().detach()[:, 1] != 4
        beta_observed_mask = beta_input.clone().detach()[:, 1] != 4
        peptide_observed_mask = peptide_input.clone().detach()[:, 1] != 4

        clf_label = binder.clone()
        labels = clf_label

        out = model(
            input_ids=(alpha_input, beta_input, peptide_input),
            attention_mask=(alpha_mask, beta_mask, peptide_mask),
            labels=labels,
            mhc=mhc
        )

        prediction_scoresA = out.decoder_outputsA.lm_logits
        predictionsA = F.log_softmax(prediction_scoresA, dim=2)
        prediction_scoresB = out.decoder_outputsB.lm_logits
        predictionsB = F.log_softmax(prediction_scoresB, dim=2)
        prediction_scoresE = out.decoder_outputsE.lm_logits
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)

        lossa = compute_loss(predictionsA[alpha_observed_mask], alpha_input[alpha_observed_mask], criterion)
        lossb = compute_loss(predictionsB[beta_observed_mask], beta_input[beta_observed_mask], criterion)
        losse = compute_loss(predictionsE[peptide_observed_mask], peptide_input[peptide_observed_mask], criterion)

        mlm_lossA = MLM_Loss(model.encoderA, model.MLMHeadA, masker, alpha_input, alpha_mask, alpha_observed_mask)
        mlm_lossB = MLM_Loss(model.encoderB, model.MLMHeadB, masker, beta_input, beta_mask, beta_observed_mask)
        mlm_lossE = MLM_Loss(model.encoderE, model.MLMHeadE, masker, peptide_input, peptide_mask, peptide_observed_mask)

        loss = mlm_lossA + mlm_lossB + mlm_lossE + (lossa + lossb + losse)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        count_A += sum(alpha_observed_mask)
        count_B += sum(beta_observed_mask)
        count_E += sum(peptide_observed_mask)
        epoch_lm_lossA += lossa if isinstance(lossa, (int, float)) else lossa.item()
        epoch_lm_lossB += lossb if isinstance(lossb, (int, float)) else lossb.item()
        epoch_lm_lossE += losse if isinstance(losse, (int, float)) else losse.item()
        epoch_mlm_lossA += mlm_lossA if isinstance(mlm_lossA, (int, float)) else mlm_lossA.item()
        epoch_mlm_lossB += mlm_lossB if isinstance(mlm_lossB, (int, float)) else mlm_lossB.item()
        epoch_mlm_lossE += mlm_lossE if isinstance(mlm_lossE, (int, float)) else mlm_lossE.item()

    count_A = max(count_A, 1)
    count_B = max(count_B, 1)
    count_E = max(count_E, 1)

    epoch_lm_lossA /= count_A
    epoch_lm_lossB /= count_B
    epoch_lm_lossE /= count_E
    epoch_mlm_lossA /= count_A
    epoch_mlm_lossB /= count_B
    epoch_mlm_lossE /= count_E

    return epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE


def evaluate_tulip(
    model,
    test_df: pd.DataFrame,
    tokenizer,
    mhctok,
    device: torch.device,
    batch_size: int = 100,
) -> Dict:
    """
    Evaluate TULIP model on test data.

    Args:
        model: TULIP model
        test_df: Test DataFrame with CDR3a, CDR3b, peptide, MHC, binder
        tokenizer: Amino acid tokenizer
        mhctok: MHC tokenizer
        device: PyTorch device
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with scores, binders, and peptides
    """
    from src.multiTrans import TCRDataset, get_logscore

    dataset = TCRDataset.from_pandas(test_df, tokenizer, device, mhctok=mhctok)

    scores = get_logscore(dataset, model, ignore_index=tokenizer.pad_token_id)
    scores = [-s for s in scores]

    return {
        "scores": np.array(scores),
        "binders": np.array(dataset.binder),
        "peptides": np.array(dataset.peptide),
    }


def compute_tulip_metrics(
    scores: np.ndarray,
    binders: np.ndarray,
    peptides: np.ndarray,
) -> Dict:
    """
    Compute per-peptide AUC-ROC and aggregated metrics.
    """
    unique_peptides = np.unique(peptides)

    per_peptide_auc = {}
    auc_values = []
    sample_counts = []

    for peptide in unique_peptides:
        mask = peptides == peptide
        pep_scores = scores[mask]
        pep_binders = binders[mask]

        if len(np.unique(pep_binders)) < 2:
            continue

        try:
            auc = roc_auc_score(pep_binders, pep_scores)
            per_peptide_auc[peptide] = {
                "auc_roc": float(auc),
                "n_samples": int(len(pep_scores)),
                "n_positive": int(sum(pep_binders)),
                "n_negative": int(len(pep_binders) - sum(pep_binders)),
            }
            auc_values.append(auc)
            sample_counts.append(len(pep_scores))
        except ValueError:
            continue

    if len(auc_values) > 0:
        mean_auc = float(np.mean(auc_values))
        weighted_mean_auc = float(np.average(auc_values, weights=sample_counts))
    else:
        mean_auc = None
        weighted_mean_auc = None

    return {
        "mean_auc_roc": mean_auc,
        "weighted_mean_auc_roc": weighted_mean_auc,
        "num_peptides_evaluated": len(auc_values),
        "num_peptides_total": len(unique_peptides),
        "num_samples": int(len(scores)),
        "per_peptide_metrics": per_peptide_auc,
    }


def compute_retrieval_scores_tulip(
    model,
    positive_df: pd.DataFrame,
    tokenizer,
    mhctok,
    device: torch.device,
    batch_size: int = 100,
) -> Dict:
    """
    Compute retrieval score matrix for TULIP.

    For each positive sample, scores the sample's TCR against ALL unique
    peptides in the test split to build a (n_samples, n_candidates) matrix.

    Args:
        model: TULIP model
        positive_df: DataFrame with positive samples (CDR3a, CDR3b, peptide, MHC)
        tokenizer: Amino acid tokenizer
        mhctok: MHC tokenizer
        device: PyTorch device
        batch_size: Batch size for scoring

    Returns:
        Dict with score_matrix, true_indices, sample_peptides, candidate_peptides
    """
    from src.multiTrans import TCRDataset, get_logscore

    unique_peptides = sorted(positive_df["peptide"].unique())
    peptide_to_idx = {pep: idx for idx, pep in enumerate(unique_peptides)}
    n_candidates = len(unique_peptides)
    n_samples = len(positive_df)

    print(f"    Retrieval scoring: {n_samples} samples x {n_candidates} candidates")

    # Build a DataFrame with all (sample_TCR, candidate_peptide) pairs
    retrieval_rows = []
    true_indices = []
    sample_peptides = []

    for idx in range(n_samples):
        row = positive_df.iloc[idx]
        true_peptide = row["peptide"]

        if true_peptide not in peptide_to_idx:
            continue

        true_indices.append(peptide_to_idx[true_peptide])
        sample_peptides.append(true_peptide)

        for pep in unique_peptides:
            retrieval_rows.append({
                "CDR3a": row["CDR3a"],
                "CDR3b": row["CDR3b"],
                "peptide": pep,
                "MHC": row["MHC"],
                "binder": 1 if pep == true_peptide else 0,
            })

    if len(true_indices) == 0:
        return None

    n_valid = len(true_indices)
    retrieval_df = pd.DataFrame(retrieval_rows)

    # Score all pairs
    dataset = TCRDataset.from_pandas(retrieval_df, tokenizer, device, mhctok=mhctok)
    scores = get_logscore(dataset, model, ignore_index=tokenizer.pad_token_id)
    # TULIP: higher log-likelihood = better binding, negate NLL
    scores = np.array([-s for s in scores])

    # Reshape into (n_valid, n_candidates)
    score_matrix = scores.reshape(n_valid, n_candidates)

    return {
        "score_matrix": score_matrix,
        "true_indices": np.array(true_indices),
        "sample_peptides": np.array(sample_peptides),
        "candidate_peptides": unique_peptides,
    }


def run_benchmark(
    task_name: str,
    data_dir: str,
    output_dir: str,
    tulip_root: str,
    config_path: str,
    num_epochs: int = 100,
    batch_size: int = 512,
    lr: float = 0.0001,
    weight_decay: float = 0.0,
    neg_per_pos: int = 5,
    use_mhc: bool = True,
    masking_proba: float = 0.0,
    save_model: bool = False,
    eval_every: int = 10,
    use_wandb: bool = False,
):
    """
    Run training and evaluation benchmark.

    Args:
        task_name: Name of the task
        data_dir: Base data directory
        output_dir: Output directory for results
        tulip_root: Path to TULIP-TCR root directory
        config_path: Path to model config
        num_epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: Weight decay
        neg_per_pos: Negatives per positive
        use_mhc: Whether to use MHC
        masking_proba: Chain masking probability
        save_model: Whether to save trained model
        eval_every: Evaluate every N epochs
        use_wandb: Whether to use wandb logging

    Returns:
        Results dictionary
    """
    from src.multiTrans import TCRDataset, MyMasking

    task_config = TULIP_TASK_CONFIGS[task_name]
    mhc_class = task_config["mhc_class"]
    task_data_dir = os.path.join(data_dir, mhc_class, task_name)

    mhc_suffix = "with_mhc" if use_mhc else "without_mhc"
    task_output_dir = os.path.join(output_dir, "trained", task_name, mhc_suffix)
    os.makedirs(task_output_dir, exist_ok=True)
    os.makedirs(os.path.join(task_output_dir, "predictions"), exist_ok=True)
    if save_model:
        os.makedirs(os.path.join(task_output_dir, "model_checkpoint"), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Description: {task_config['description']}")
    print(f"Use MHC: {use_mhc}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Negatives per positive: {neg_per_pos}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(0)
    np.random.seed(0)

    # Setup tokenizers
    print("\nSetting up tokenizers...")
    tokenizer, mhctok = setup_tokenizer(tulip_root, use_mhc=use_mhc)

    # Load model config
    with open(config_path, "r") as f:
        modelconfig = json.load(f)

    # Initialize model
    print("Initializing model...")
    model = initialize_tulip_model(modelconfig, tokenizer, mhctok, device)

    def count_parameters(mdl):
        return sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"Model has {count_parameters(model):,} trainable parameters")

    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    masker = MyMasking(tokenizer, mlm_probability=0.15)

    # Load training data
    print("\nLoading training data...")
    train_path = os.path.join(task_data_dir, "train.parquet")
    train_df = convert_parquet_to_tulip_df(train_path, task_config, use_mhc=use_mhc)
    print(f"Training positives: {len(train_df)}")

    # Generate negatives for training
    # Note: For training, we include all samples as positives since TULIP learns generatively
    # The negatives are mainly for evaluation
    # However, we can still add some negatives to help with discrimination
    if neg_per_pos > 0:
        train_df_with_neg = generate_negatives(train_df.copy(), neg_per_pos=neg_per_pos)
        print(f"Training with negatives: {len(train_df_with_neg)}")
    else:
        train_df_with_neg = train_df.copy()

    # Create dataset - for training, use only positives (TULIP is generative)
    train_dataset = TCRDataset.from_pandas(train_df, tokenizer, device, mhctok=mhctok)
    train_dataset.set_chain_masking_proba(proba=masking_proba)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.all2allmhc_collate_function
    )

    # Load validation data for periodic evaluation
    print("Loading validation data...")
    val_path = os.path.join(task_data_dir, "val.parquet")
    val_positive_df = convert_parquet_to_tulip_df(val_path, task_config, use_mhc=use_mhc)
    val_df = generate_negatives(val_positive_df.copy(), neg_per_pos=neg_per_pos)
    print(f"Validation samples: {len(val_df)}")

    # Initialize wandb if requested
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="tulip-benchmark",
                config={
                    'task': task_name,
                    'batch_size': batch_size,
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'num_epochs': num_epochs,
                    'use_mhc': use_mhc,
                    'neg_per_pos': neg_per_pos,
                    **modelconfig
                }
            )
        except ImportError:
            print("wandb not installed, skipping logging")
            use_wandb = False

    # Training loop
    training_log = []
    best_val_auc = 0.0
    best_epoch = 0

    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Train
        losses = train_one_epoch(
            model, optimizer, train_dataloader, criterion, masker, device, masking_proba
        )
        epoch_lm_lossA, epoch_lm_lossB, epoch_lm_lossE, epoch_mlm_lossA, epoch_mlm_lossB, epoch_mlm_lossE = losses

        log_entry = {
            "epoch": epoch,
            "lm_loss_A": float(epoch_lm_lossA),
            "lm_loss_B": float(epoch_lm_lossB),
            "lm_loss_E": float(epoch_lm_lossE),
            "mlm_loss_A": float(epoch_mlm_lossA),
            "mlm_loss_B": float(epoch_mlm_lossB),
            "mlm_loss_E": float(epoch_mlm_lossE),
        }

        # Evaluate periodically
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            eval_results = evaluate_tulip(model, val_df, tokenizer, mhctok, device)
            metrics = compute_tulip_metrics(
                eval_results["scores"], eval_results["binders"], eval_results["peptides"]
            )

            log_entry["val_mean_auc"] = metrics["mean_auc_roc"]
            log_entry["val_weighted_auc"] = metrics["weighted_mean_auc_roc"]

            if metrics["mean_auc_roc"] and metrics["mean_auc_roc"] > best_val_auc:
                best_val_auc = metrics["mean_auc_roc"]
                best_epoch = epoch
                if save_model:
                    model.save_pretrained(os.path.join(task_output_dir, "model_checkpoint", "best"))

            print(f"Epoch {epoch}: LM Loss (A/B/E): {epoch_lm_lossA:.4f}/{epoch_lm_lossB:.4f}/{epoch_lm_lossE:.4f}, "
                  f"Val AUC: {metrics['mean_auc_roc']:.4f}" if metrics['mean_auc_roc'] else f"Epoch {epoch}: LM Loss (A/B/E): {epoch_lm_lossA:.4f}/{epoch_lm_lossB:.4f}/{epoch_lm_lossE:.4f}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: LM Loss (A/B/E): {epoch_lm_lossA:.4f}/{epoch_lm_lossB:.4f}/{epoch_lm_lossE:.4f}")

        training_log.append(log_entry)

        if use_wandb:
            wandb.log(log_entry)

        # Save checkpoint
        if save_model and epoch % 50 == 0 and epoch > 0:
            model.save_pretrained(os.path.join(task_output_dir, "model_checkpoint", f"epoch_{epoch}"))

    # Save final model
    if save_model:
        model.save_pretrained(os.path.join(task_output_dir, "model_checkpoint", "final"))

    # Save training log
    log_path = os.path.join(task_output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    # Results
    results = {
        "task": task_name,
        "use_mhc": use_mhc,
        "neg_per_pos": neg_per_pos,
        "config_path": config_path,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "test_results": {},
    }

    # Evaluate on all test splits
    print("\nEvaluating on test splits...")
    for split_name in TEST_SPLITS:
        test_path = os.path.join(task_data_dir, f"{split_name}.parquet")
        if not os.path.exists(test_path):
            print(f"  Skipping {split_name} - file not found")
            continue

        print(f"\n  Processing {split_name}...")

        positive_df = convert_parquet_to_tulip_df(test_path, task_config, use_mhc=use_mhc)
        print(f"    Positive samples: {len(positive_df)}")

        test_df = generate_negatives(positive_df, neg_per_pos=neg_per_pos)
        print(f"    Total samples: {len(test_df)}")

        eval_results = evaluate_tulip(model, test_df, tokenizer, mhctok, device)
        metrics = compute_tulip_metrics(
            eval_results["scores"], eval_results["binders"], eval_results["peptides"]
        )

        print(f"    Mean AUC-ROC: {metrics['mean_auc_roc']:.4f}" if metrics['mean_auc_roc'] else "    Mean AUC-ROC: N/A")
        print(f"    Peptides evaluated: {metrics['num_peptides_evaluated']}/{metrics['num_peptides_total']}")

        # Compute unified retrieval metrics
        print(f"    Computing retrieval metrics...")
        retrieval_data = compute_retrieval_scores_tulip(
            model, positive_df, tokenizer, mhctok, device, batch_size=batch_size
        )

        if retrieval_data is not None:
            unified = compute_all_unified_metrics(
                scores=retrieval_data["score_matrix"],
                true_indices=retrieval_data["true_indices"],
                sample_peptides=retrieval_data["sample_peptides"],
                candidate_peptides=retrieval_data["candidate_peptides"],
                n_bootstrap=1000,
                min_samples_per_peptide=5,
                seed=42,
            )

            per_epitope_all = unified.pop("per_epitope_all", [])
            metrics.update(unified)

            print(f"    Retrieval Hit@1: {metrics['retrieval_hit_at_1']:.4f}")
            print(f"    Retrieval MRR: {metrics['retrieval_mrr']:.4f}")
            if metrics.get("per_peptide_auc_mean") is not None:
                print(f"    Per-peptide AUC (unified): {metrics['per_peptide_auc_mean']:.4f}")

            # Save per-epitope breakdown CSV
            if per_epitope_all:
                epitope_df = pd.DataFrame(per_epitope_all)
                epitope_df.to_csv(
                    os.path.join(task_output_dir, "predictions", f"{split_name}_per_epitope_breakdown.csv"),
                    index=False,
                )

        results["test_results"][split_name] = metrics

        # Save predictions
        pred_df = pd.DataFrame({
            "CDR3a": test_df["CDR3a"],
            "CDR3b": test_df["CDR3b"],
            "peptide": test_df["peptide"],
            "MHC": test_df["MHC"],
            "binder": eval_results["binders"],
            "score": eval_results["scores"],
        })

        # Add retrieval columns for positive samples
        if retrieval_data is not None:
            # Build a mapping from positive sample index to retrieval rank
            pos_mask = test_df["binder"] == 1
            retrieval_ranks = np.zeros(len(test_df), dtype=float)
            retrieval_ranks[:] = np.nan

            score_matrix = retrieval_data["score_matrix"]
            true_idx_arr = retrieval_data["true_indices"]
            pos_idx = 0
            for i in range(len(test_df)):
                if pos_mask.iloc[i] and pos_idx < len(true_idx_arr):
                    sorted_indices = np.argsort(-score_matrix[pos_idx])
                    rank = int(np.where(sorted_indices == true_idx_arr[pos_idx])[0][0]) + 1
                    retrieval_ranks[i] = rank
                    pos_idx += 1

            pred_df["true_peptide_rank"] = retrieval_ranks

        pred_path = os.path.join(task_output_dir, "predictions", f"{split_name}_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

    # Save results
    results_path = os.path.join(task_output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    if use_wandb:
        wandb.finish()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate TULIP model on TCR-peptide binding prediction"
    )
    parser.add_argument(
        "--task",
        choices=list(TULIP_TASK_CONFIGS.keys()) + ["all"],
        default="all",
        help="Task to run (default: all)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/sravisha/projects/quest/data/icml/tasks/tcr_specificity/tcr90pep80",
        help="Base data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sravisha/projects/quest/results/tulip_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--tulip_root",
        type=str,
        default=TULIP_ROOT,
        help="Path to TULIP-TCR root directory",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to model config (default: tulip_root/configs/shallow.config.json)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Training batch size (default: 512)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (default: 0.0)",
    )
    parser.add_argument(
        "--neg_per_pos",
        type=int,
        default=5,
        help="Negatives per positive (default: 5)",
    )
    parser.add_argument(
        "--skip_mhc",
        action="store_true",
        help="Skip MHC information (use nomhc tokenizer)",
    )
    parser.add_argument(
        "--masking_proba",
        type=float,
        default=0.0,
        help="Chain masking probability during training (default: 0.0)",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save trained model checkpoints",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=10,
        help="Evaluate every N epochs (default: 10)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use wandb for logging",
    )
    args = parser.parse_args()

    # Set default config path
    if args.config_path is None:
        args.config_path = os.path.join(args.tulip_root, "configs/shallow.config.json")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine tasks
    tasks = [args.task] if args.task != "all" else list(TULIP_TASK_CONFIGS.keys())

    # Track all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config_path": args.config_path,
        "use_mhc": not args.skip_mhc,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "neg_per_pos": args.neg_per_pos,
        "tasks": {},
    }

    # Run benchmarks
    for task in tasks:
        try:
            results = run_benchmark(
                task_name=task,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                tulip_root=args.tulip_root,
                config_path=args.config_path,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                neg_per_pos=args.neg_per_pos,
                use_mhc=not args.skip_mhc,
                masking_proba=args.masking_proba,
                save_model=args.save_model,
                eval_every=args.eval_every,
                use_wandb=args.use_wandb,
            )
            all_results["tasks"][task] = results
        except Exception as e:
            print(f"\nError running {task}: {e}")
            import traceback
            traceback.print_exc()
            all_results["tasks"][task] = {"error": str(e)}

    # Save summary
    summary_path = os.path.join(args.output_dir, "trained", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Benchmark complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
