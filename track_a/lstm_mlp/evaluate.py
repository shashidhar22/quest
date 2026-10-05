"""Retrieval evaluation for LSTM/MLP AS baselines.

For each test row: score (TCR_test, peptide_candidate, MHC_test) for every
peptide in the candidate pool; rank candidates; locate the true peptide's
rank. Compute Recall@K, MRR, Hit@1.

Writes results JSON in the user-spec format and a per-(query, candidate)
score sidecar parquet.

Usage:
    python -m track_a.lstm_mlp.evaluate \\
        --task as_trb_i \\
        --checkpoint /path/to/best.pt \\
        --baseline_name lstm \\
        [--smoke]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from track_a.lstm_mlp.convert_data import load_candidates, load_test_condition
from track_a.lstm_mlp.dataset import (
    DEFAULT_MAX_PEP, DEFAULT_MAX_TRA, DEFAULT_MAX_TRB,
    TripletDataset, collate, encode_allele, one_hot_mhc,
)
from track_a.lstm_mlp.model import TCRPeptideMHCScorer, encode_seq
from track_a.shared.load_splits import (
    CLASS_I_REPORT_CONDITIONS,
    CLASS_I_TASKS,
    CLASS_II_TASKS,
    all_test_files,
    class_i_companion,
)
from track_a.shared.results_schema import write_results, write_scores_sidecar


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[TCRPeptideMHCScorer, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TCRPeptideMHCScorer(
        encoder_type=ckpt["encoder_type"],
        chains=ckpt["chains"],
        mhc_dim=ckpt["mhc_dim"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _score_against_pool(
    model: TCRPeptideMHCScorer,
    test_df: pd.DataFrame,
    candidates: list[str],
    allele_vocab: dict,
    task: str,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores [n_test, n_cand], true_idx [n_test]).

    For each test row we score the same TRB/MHC against every candidate
    peptide. We pre-encode the TRB / TRA / MHC once per row (broadcast
    across candidates), and pre-encode all candidate peptides once.
    """
    chains = model.chains
    mhc_col = "mhc_one_allele" if task.endswith("_i") else "mhc_two_allele"
    n_cand = len(candidates)

    # Map true peptide → candidate index. If a test row's true peptide is
    # not in the pool we record -1 (excluded from metrics).
    cand_to_idx = {p: i for i, p in enumerate(candidates)}
    true_idx = np.array([cand_to_idx.get(str(p), -1) for p in test_df["peptide"].values])

    # Pre-encode candidates (n_cand, max_pep)
    cand_ids = torch.stack([encode_seq(c, DEFAULT_MAX_PEP)[0] for c in candidates], dim=0).to(device)
    cand_lens = torch.tensor([encode_seq(c, DEFAULT_MAX_PEP)[1] for c in candidates], dtype=torch.long).to(device)

    scores = np.zeros((len(test_df), n_cand), dtype=np.float32)

    for start in range(0, len(test_df), batch_size):
        end = min(start + batch_size, len(test_df))
        sub = test_df.iloc[start:end]
        # Encode TRB, TRA (if paired), MHC for the sub-batch (one row each)
        trb_ids = torch.stack([encode_seq(s, DEFAULT_MAX_TRB)[0] for s in sub["trb_cdr3"].values], dim=0).to(device)
        trb_lens = torch.tensor([encode_seq(s, DEFAULT_MAX_TRB)[1] for s in sub["trb_cdr3"].values], dtype=torch.long).to(device)
        if chains == "paired":
            tra_vals = sub["tra_cdr3"].values
            tra_ids = torch.stack([encode_seq(s, DEFAULT_MAX_TRA)[0] for s in tra_vals], dim=0).to(device)
            tra_lens = torch.tensor([encode_seq(s, DEFAULT_MAX_TRA)[1] for s in tra_vals], dtype=torch.long).to(device)
        else:
            tra_ids, tra_lens = None, None
        allele_indices = [encode_allele(a, allele_vocab) for a in sub[mhc_col].values]
        mhc_oh = torch.stack([one_hot_mhc(i, len(allele_vocab)) for i in allele_indices], dim=0).to(device)

        # For each row in sub, score against all candidates: replicate row
        # features and tile candidate features.
        n_sub = len(sub)
        # (n_sub, n_cand) outputs. Use double loop over sub rows (n_cand
        # is ~3.5k; one row at a time keeps memory bounded).
        with torch.no_grad():
            for i in range(n_sub):
                trb_i = trb_ids[i:i+1].expand(n_cand, -1)
                trb_l_i = trb_lens[i:i+1].expand(n_cand)
                mhc_i = mhc_oh[i:i+1].expand(n_cand, -1)
                if chains == "paired":
                    tra_i = tra_ids[i:i+1].expand(n_cand, -1)
                    tra_l_i = tra_lens[i:i+1].expand(n_cand)
                    logits = model(trb_ids=trb_i, trb_len=trb_l_i,
                                   pep_ids=cand_ids, pep_len=cand_lens,
                                   mhc_oh=mhc_i,
                                   tra_ids=tra_i, tra_len=tra_l_i)
                else:
                    logits = model(trb_ids=trb_i, trb_len=trb_l_i,
                                   pep_ids=cand_ids, pep_len=cand_lens,
                                   mhc_oh=mhc_i)
                scores[start + i] = logits.cpu().numpy()
    return scores, true_idx


def _retrieval_metrics(scores: np.ndarray, true_idx: np.ndarray) -> dict:
    valid = true_idx >= 0
    if not valid.any():
        return {"recall_at_10": 0.0, "mrr": 0.0, "hit_at_1": 0.0,
                "n_evaluated": 0, "n_excluded_no_candidate": int((~valid).sum())}
    s = scores[valid]
    t = true_idx[valid]
    # rank of true (1-indexed)
    ranks = np.empty(len(t), dtype=np.int64)
    for i in range(len(t)):
        # arg-rank where higher score = better; use stable sort
        order = np.argsort(-s[i], kind="stable")
        ranks[i] = int(np.where(order == t[i])[0][0]) + 1
    return {
        "recall_at_10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "hit_at_1": float((ranks == 1).mean()),
        "n_evaluated": int(valid.sum()),
        "n_excluded_no_candidate": int((~valid).sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="as_trb_i / as_paired_i / as_trb_ii / as_paired_ii")
    p.add_argument("--checkpoint", required=True, help="best.pt from train.py")
    p.add_argument("--baseline_name", required=True, help="e.g. 'lstm', 'mlp'")
    p.add_argument("--results_root", default="/home/ubuntu/quest/results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_n", type=int, default=50)
    p.add_argument("--max_test_rows", type=int, default=0, help="0=all; otherwise cap test rows for speed")
    p.add_argument("--wandb_run_id", default=None)
    args = p.parse_args()

    device = torch.device(args.device)

    print(f"[lstm_mlp/eval] task={args.task} ckpt={args.checkpoint} device={device}", flush=True)
    model, ckpt = _load_model(Path(args.checkpoint), device)
    allele_vocab = ckpt["allele_vocab"]
    train_task = args.task if args.task in CLASS_I_TASKS else class_i_companion(args.task)
    training_data = f"benchmark_v2/splits/{train_task}_train.parquet"

    candidates = load_candidates(args.task)
    print(f"[lstm_mlp/eval]  candidate pool ({args.task}): {len(candidates)} peptides", flush=True)

    test_files = all_test_files(args.task)
    conditions_out: dict[str, dict[str, float]] = {}
    for cond, _path in test_files.items():
        t0 = time.time()
        test_df = load_test_condition(args.task, cond)
        if args.smoke:
            test_df = test_df.head(args.smoke_n)
        elif args.max_test_rows:
            test_df = test_df.head(args.max_test_rows)
        # Drop rows with no peptide / TRB
        test_df = test_df[test_df["peptide"].notna() & test_df["trb_cdr3"].notna()].reset_index(drop=True)
        print(f"[lstm_mlp/eval]  {cond}: {len(test_df)} test rows", flush=True)
        if len(test_df) == 0:
            conditions_out[cond] = {"recall_at_10": 0.0, "mrr": 0.0, "hit_at_1": 0.0, "n_evaluated": 0}
            continue
        scores, true_idx = _score_against_pool(
            model=model, test_df=test_df, candidates=candidates,
            allele_vocab=allele_vocab, task=args.task, device=device,
            batch_size=args.batch_size,
        )
        # Score sidecar (skip on smoke to save time)
        if not args.smoke:
            write_scores_sidecar(
                baseline=args.baseline_name, task=args.task, condition=cond,
                query_ids=test_df["row_id"].astype(str).values,
                candidate_peptides=candidates,
                scores=scores,
                true_peptides=test_df["peptide"].astype(str).values,
                out_root=Path(args.results_root),
            )
        metrics = _retrieval_metrics(scores, true_idx)
        elapsed = time.time() - t0
        metrics["elapsed_secs"] = round(elapsed, 1)
        conditions_out[cond] = metrics
        print(f"[lstm_mlp/eval]   {cond}: recall@10={metrics['recall_at_10']:.4f} mrr={metrics['mrr']:.4f} hit@1={metrics['hit_at_1']:.4f}  ({elapsed:.1f}s)", flush=True)

    write_results(
        baseline=args.baseline_name,
        task=args.task,
        conditions=conditions_out,
        wandb_run_id=args.wandb_run_id,
        training_data=training_data,
        out_root=Path(args.results_root),
    )
    print(f"[lstm_mlp/eval] wrote results JSON for {args.baseline_name}/{args.task}", flush=True)


if __name__ == "__main__":
    main()
