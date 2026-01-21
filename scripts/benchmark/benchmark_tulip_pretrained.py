#!/usr/bin/env python
"""
TULIP-TCR Benchmark Script - Pretrained Model Evaluation

This script evaluates the pretrained TULIP model on TCR-peptide binding prediction tasks.
TULIP uses a generative approach where binding affinity is scored by the negative log-likelihood
of generating the peptide sequence given the TCR and MHC context.

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

# Add TULIP-TCR to path
TULIP_ROOT = "/home/sravisha/tcrbench_tools/TULIP-TCR"
sys.path.insert(0, TULIP_ROOT)

# Suppress warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
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
        # Class I: Single allele like 'A*02:01' -> 'HLA-A*02:01'
        if not mhc_id.startswith('HLA-'):
            return f"HLA-{mhc_id}"
        return mhc_id
    else:
        # Class II: Combined format like 'DRA*01:01_DRB1*01:01' -> 'HLA-DRA*01:01/DRB1*01:01'
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
            # For MHC-II, combine alpha and beta chains
            col1 = df[mhc_cols[0]].fillna('')
            col2 = df[mhc_cols[1]].fillna('')
            mhc_values = col1.astype(str) + "_" + col2.astype(str)

        tulip_df["MHC"] = mhc_values.apply(lambda x: format_mhc_for_tulip(x, mhc_class))
    else:
        # Set all MHC to <MIS> when not using MHC information
        tulip_df["MHC"] = '<MIS>'

    # Binder column
    if add_binder_column:
        tulip_df["binder"] = 1  # All samples in our data are positive binders

    # Store original peptide for grouping
    tulip_df["original_peptide"] = df["peptide"]

    return tulip_df


def generate_negatives(positive_df: pd.DataFrame, neg_per_pos: int = 5) -> pd.DataFrame:
    """
    Generate negative samples using TULIP paper approach.

    For each positive TCR-peptide pair, generate N negatives by sampling TCRs
    from other peptides and pairing with the current peptide.
    Ensure sampled TCRs don't bind target peptide.

    Args:
        positive_df: DataFrame with positive samples (CDR3a, CDR3b, peptide, MHC, binder)
        neg_per_pos: Number of negatives to generate per positive

    Returns:
        Combined DataFrame with positives and negatives
    """
    peptides = positive_df["peptide"].unique()

    # Build lookup of TCRs that bind each peptide
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
        max_attempts = neg_per_pos * 10  # Avoid infinite loops
        attempts = 0

        while neg_count < neg_per_pos and attempts < max_attempts:
            attempts += 1

            # Sample a random row from a different peptide
            sample_idx = np.random.randint(len(positive_df))
            sample_row = positive_df.iloc[sample_idx]

            if sample_row["peptide"] == target_peptide:
                continue

            sample_alpha = sample_row["CDR3a"]
            sample_beta = sample_row["CDR3b"]

            # Check if this TCR is known to bind the target peptide
            is_alpha_binder = (sample_alpha != '<MIS>' and
                              sample_alpha in peptide_alpha.get(target_peptide, set()))
            is_beta_binder = (sample_beta != '<MIS>' and
                             sample_beta in peptide_beta.get(target_peptide, set()))

            # Skip if either chain is known to bind target peptide
            if is_alpha_binder or is_beta_binder:
                continue

            # Create negative sample
            negatives.append({
                "CDR3a": sample_alpha,
                "CDR3b": sample_beta,
                "peptide": target_peptide,
                "MHC": target_mhc,
                "binder": 0,
                "original_peptide": target_peptide,
            })
            neg_count += 1

    # Combine positives and negatives
    neg_df = pd.DataFrame(negatives)
    combined_df = pd.concat([positive_df, neg_df], ignore_index=True)

    return combined_df


def setup_tokenizer(tulip_root: str, use_mhc: bool = True):
    """
    Set up TULIP tokenizers.

    Args:
        tulip_root: Path to TULIP-TCR root directory
        use_mhc: Whether to use MHC information (if False, MHC will be set to <MIS>)
                 Note: For pretrained models, we always use mhctok to match the model architecture

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

    # Always use mhctok for pretrained models to match the model's MHC vocabulary
    # The use_mhc flag controls whether we mask MHC values in the data, not the tokenizer
    mhctok = AutoTokenizer.from_pretrained(os.path.join(tulip_root, "mhctok/"))

    return tokenizer, mhctok


def load_pretrained_tulip(
    tulip_root: str,
    model_path: str,
    config_path: str,
    tokenizer,
    mhctok,
    device: torch.device,
):
    """
    Load pretrained TULIP model.

    Args:
        tulip_root: Path to TULIP-TCR root directory
        model_path: Path to model weights (pytorch_model.bin)
        config_path: Path to model config JSON
        tokenizer: Amino acid tokenizer
        mhctok: MHC tokenizer (used for data, but model always uses full mhctok vocab)
        device: PyTorch device

    Returns:
        Loaded TULIP model
    """
    from src.multiTrans import TulipPetal, BertLastPooler, Tulip

    with open(config_path, "r") as f:
        modelconfig = json.load(f)

    vocabsize = len(tokenizer._tokenizer.get_vocab())

    # Always use the full mhctok vocabulary size for model architecture
    # The pretrained model was trained with mhctok (55 tokens), so we must match it
    full_mhctok = AutoTokenizer.from_pretrained(os.path.join(tulip_root, "mhctok/"))
    mhcvocabsize = len(full_mhctok._tokenizer.get_vocab())

    max_length = 50

    # Check if config has decoupled architecture
    if "num_attn_heads_encoder_cdr" in modelconfig:
        # Decoupled config
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
        # Shared config
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

    # Load weights
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.to(device)

    return model


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

    # Create dataset from DataFrame
    dataset = TCRDataset.from_pandas(test_df, tokenizer, device, mhctok=mhctok)

    # Get log scores (negative log likelihood - lower is better for binding)
    scores = get_logscore(dataset, model, ignore_index=tokenizer.pad_token_id)

    # Negate scores so higher = better binding (consistent with AUC computation)
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

    Args:
        scores: Binding scores (higher = better)
        binders: Binary binder labels (1 = positive, 0 = negative)
        peptides: Peptide sequences

    Returns:
        Dictionary with metrics
    """
    unique_peptides = np.unique(peptides)

    per_peptide_auc = {}
    auc_values = []
    sample_counts = []

    for peptide in unique_peptides:
        mask = peptides == peptide
        pep_scores = scores[mask]
        pep_binders = binders[mask]

        # Need both positive and negative samples
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
        except ValueError as e:
            continue

    # Aggregate metrics
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


def run_benchmark(
    task_name: str,
    data_dir: str,
    output_dir: str,
    tulip_root: str,
    model_path: str,
    config_path: str,
    neg_per_pos: int = 5,
    use_mhc: bool = True,
    batch_size: int = 100,
):
    """
    Run benchmark for a single task.

    Args:
        task_name: Name of the task
        data_dir: Base data directory
        output_dir: Output directory for results
        tulip_root: Path to TULIP-TCR root directory
        model_path: Path to model weights
        config_path: Path to model config
        neg_per_pos: Number of negatives per positive
        use_mhc: Whether to use MHC information
        batch_size: Batch size for evaluation

    Returns:
        Results dictionary
    """
    task_config = TULIP_TASK_CONFIGS[task_name]
    mhc_class = task_config["mhc_class"]
    task_data_dir = os.path.join(data_dir, mhc_class, task_name)

    # Create output directory
    mhc_suffix = "with_mhc" if use_mhc else "without_mhc"
    task_output_dir = os.path.join(output_dir, "pretrained", task_name, mhc_suffix)
    os.makedirs(task_output_dir, exist_ok=True)
    os.makedirs(os.path.join(task_output_dir, "predictions"), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Description: {task_config['description']}")
    print(f"Use MHC: {use_mhc}")
    print(f"Negatives per positive: {neg_per_pos}")
    print(f"{'='*60}")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup tokenizers
    print("\nSetting up tokenizers...")
    tokenizer, mhctok = setup_tokenizer(tulip_root, use_mhc=use_mhc)

    # Load model
    print("Loading pretrained model...")
    model = load_pretrained_tulip(
        tulip_root, model_path, config_path, tokenizer, mhctok, device
    )
    model.eval()

    # Results
    results = {
        "task": task_name,
        "use_mhc": use_mhc,
        "neg_per_pos": neg_per_pos,
        "model_path": model_path,
        "config_path": config_path,
        "test_results": {},
    }

    # Evaluate on each test split
    print("\nEvaluating on test splits...")
    for split_name in TEST_SPLITS:
        test_path = os.path.join(task_data_dir, f"{split_name}.parquet")
        if not os.path.exists(test_path):
            print(f"  Skipping {split_name} - file not found")
            continue

        print(f"\n  Processing {split_name}...")

        # Convert parquet to TULIP format
        positive_df = convert_parquet_to_tulip_df(test_path, task_config, use_mhc=use_mhc)
        print(f"    Positive samples: {len(positive_df)}")

        # Generate negatives
        test_df = generate_negatives(positive_df, neg_per_pos=neg_per_pos)
        print(f"    Total samples (with negatives): {len(test_df)}")

        # Evaluate
        eval_results = evaluate_tulip(
            model, test_df, tokenizer, mhctok, device, batch_size
        )

        # Compute metrics
        metrics = compute_tulip_metrics(
            eval_results["scores"],
            eval_results["binders"],
            eval_results["peptides"],
        )

        results["test_results"][split_name] = metrics

        print(f"    Mean AUC-ROC: {metrics['mean_auc_roc']:.4f}" if metrics['mean_auc_roc'] else "    Mean AUC-ROC: N/A")
        print(f"    Weighted Mean AUC-ROC: {metrics['weighted_mean_auc_roc']:.4f}" if metrics['weighted_mean_auc_roc'] else "    Weighted Mean AUC-ROC: N/A")
        print(f"    Peptides evaluated: {metrics['num_peptides_evaluated']}/{metrics['num_peptides_total']}")

        # Save predictions
        pred_df = pd.DataFrame({
            "CDR3a": test_df["CDR3a"],
            "CDR3b": test_df["CDR3b"],
            "peptide": test_df["peptide"],
            "MHC": test_df["MHC"],
            "binder": eval_results["binders"],
            "score": eval_results["scores"],
        })
        pred_path = os.path.join(task_output_dir, "predictions", f"{split_name}_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

    # Save results
    results_path = os.path.join(task_output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pretrained TULIP model on TCR-peptide binding prediction"
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
        "--model_path",
        type=str,
        default=None,
        help="Path to model weights (default: tulip_root/model_weights/pytorch_model.bin)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to model config (default: tulip_root/configs/shallow.config.json)",
    )
    parser.add_argument(
        "--neg_per_pos",
        type=int,
        default=5,
        help="Number of negative samples per positive (default: 5)",
    )
    parser.add_argument(
        "--skip_mhc",
        action="store_true",
        help="Skip MHC information (use nomhc tokenizer)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for evaluation (default: 100)",
    )
    args = parser.parse_args()

    # Set default paths
    if args.model_path is None:
        args.model_path = os.path.join(args.tulip_root, "model_weights/pytorch_model.bin")
    if args.config_path is None:
        args.config_path = os.path.join(args.tulip_root, "configs/shallow.config.json")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine tasks to run
    tasks = [args.task] if args.task != "all" else list(TULIP_TASK_CONFIGS.keys())

    # Track all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model_path,
        "config_path": args.config_path,
        "use_mhc": not args.skip_mhc,
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
                model_path=args.model_path,
                config_path=args.config_path,
                neg_per_pos=args.neg_per_pos,
                use_mhc=not args.skip_mhc,
                batch_size=args.batch_size,
            )
            all_results["tasks"][task] = results
        except Exception as e:
            print(f"\nError running {task}: {e}")
            import traceback
            traceback.print_exc()
            all_results["tasks"][task] = {"error": str(e)}

    # Save summary
    summary_path = os.path.join(args.output_dir, "pretrained", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Benchmark complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
