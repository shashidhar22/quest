"""Retrieval evaluation for ESM SFT cells."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from track_a.esm_sft.convert_data import load_candidates, load_test_condition
from track_a.esm_sft.dataset import assemble_sequence
from track_a.esm_sft.encoders import LoadedEncoder, load_encoder
from track_a.esm_sft.model import ESMScorer
from track_a.shared.load_splits import (
    CLASS_I_TASKS, CLASS_II_TASKS, all_test_files, class_i_companion,
)
from track_a.shared.results_schema import write_results, write_scores_sidecar


def _tokenize_one(seq: str, tokenizer, is_esmc: bool, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    if is_esmc:
        ids = tokenizer.encode(seq)
        ids = torch.as_tensor(ids, dtype=torch.long)[: max_len]
        attn = torch.ones_like(ids)
    else:
        enc = tokenizer(seq, max_length=max_len, truncation=True, padding=False, return_tensors="pt")
        ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
    return ids, attn


def _pad_batch(seqs: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int):
    max_l = max(s[0].size(0) for s in seqs)
    n = len(seqs)
    ids = torch.full((n, max_l), pad_id, dtype=torch.long)
    attn = torch.zeros((n, max_l), dtype=torch.long)
    for i, (s, a) in enumerate(seqs):
        l = s.size(0)
        ids[i, :l] = s
        attn[i, :l] = a
    return ids, attn


def _retrieval(scores: np.ndarray, true_idx: np.ndarray) -> dict:
    valid = true_idx >= 0
    if not valid.any():
        return {"recall_at_10": 0.0, "mrr": 0.0, "hit_at_1": 0.0, "n_evaluated": 0,
                "n_excluded_no_candidate": int((~valid).sum())}
    s = scores[valid]; t = true_idx[valid]
    ranks = np.empty(len(t), dtype=np.int64)
    for i in range(len(t)):
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
    p.add_argument("--encoder", choices=["esm2_8m", "esm2_35m", "esmc_300m"], required=True)
    p.add_argument("--variant", choices=["base", "cpt"], required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--baseline_name", default=None)
    p.add_argument("--results_root", default="/home/ubuntu/quest/results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_n", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_test_rows", type=int, default=0)
    p.add_argument("--wandb_run_id", default=None)
    args = p.parse_args()

    name = f"{args.encoder}-{args.variant}"
    baseline_name = args.baseline_name or name
    device = torch.device(args.device)

    print(f"[esm_sft/eval] {name} task={args.task} device={device}", flush=True)
    loaded = load_encoder(name)
    model = ESMScorer(loaded, head_dim=256, dropout=0.1, freeze_base=True).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = model.state_dict()
    state.update(ckpt["trainable_state_dict"])
    model.load_state_dict(state)
    model.eval()
    pad_id = getattr(loaded.tokenizer, "pad_token_id", None) or 1

    chains = "paired" if args.task.startswith("as_paired") else "trb"
    mhc_col = "mhc_one_allele" if args.task.endswith("_i") else "mhc_two_allele"
    pseudoseq_col = "mhc_one_pocket_contact" if args.task.endswith("_i") else "mhc_two_pocket_contact"

    train_task = args.task if args.task in CLASS_I_TASKS else class_i_companion(args.task)
    training_data = f"benchmark_v2/splits/{train_task}_train.parquet"

    candidates = load_candidates(args.task)
    print(f"[esm_sft/eval]  candidate pool ({args.task}): {len(candidates)} peptides", flush=True)

    test_files = all_test_files(args.task)
    conditions_out: dict[str, dict[str, float]] = {}

    for cond, _path in test_files.items():
        t0 = time.time()
        test_df = load_test_condition(args.task, cond)
        if args.smoke:
            test_df = test_df.head(args.smoke_n)
        elif args.max_test_rows:
            test_df = test_df.head(args.max_test_rows)
        test_df = test_df[test_df["peptide"].notna() & test_df["trb_cdr3"].notna()].reset_index(drop=True)
        print(f"[esm_sft/eval]  {cond}: {len(test_df)} rows", flush=True)
        if len(test_df) == 0:
            conditions_out[cond] = {"recall_at_10": 0.0, "mrr": 0.0, "hit_at_1": 0.0, "n_evaluated": 0}
            continue
        # Build cand_to_idx + true_idx
        cand_to_idx = {p: i for i, p in enumerate(candidates)}
        true_idx = np.array([cand_to_idx.get(str(p), -1) for p in test_df["peptide"].values])

        # Score each (test_row, candidate). For efficiency, for each test row,
        # we form a batch of (candidate_count) sequences with that row's TCR/MHC
        # held fixed and the candidate peptide swapped in. ESM forward is the
        # bottleneck; we mini-batch within the candidate dimension.
        scores = np.zeros((len(test_df), len(candidates)), dtype=np.float32)
        for ti in range(len(test_df)):
            row = test_df.iloc[ti].copy()
            seqs_for_row = []
            for cp in candidates:
                row["peptide"] = cp
                seq = assemble_sequence(row, chains=chains, mhc_col=mhc_col, pseudoseq_col=pseudoseq_col)
                ids, attn = _tokenize_one(seq, loaded.tokenizer, loaded.is_esmc, args.max_len)
                seqs_for_row.append((ids, attn))
            # Batch over candidates
            for start in range(0, len(seqs_for_row), args.batch_size):
                end = min(start + args.batch_size, len(seqs_for_row))
                ids, attn = _pad_batch(seqs_for_row[start:end], pad_id=pad_id)
                ids = ids.to(device); attn = attn.to(device)
                with torch.no_grad():
                    logits = model(ids, attn)
                scores[ti, start:end] = logits.cpu().numpy()
            if (ti + 1) % 50 == 0:
                print(f"[esm_sft/eval]    scored {ti+1}/{len(test_df)} rows", flush=True)

        if not args.smoke:
            write_scores_sidecar(
                baseline=baseline_name, task=args.task, condition=cond,
                query_ids=test_df["row_id"].astype(str).values,
                candidate_peptides=candidates,
                scores=scores,
                true_peptides=test_df["peptide"].astype(str).values,
                out_root=Path(args.results_root),
            )
        m = _retrieval(scores, true_idx)
        elapsed = time.time() - t0; m["elapsed_secs"] = round(elapsed, 1)
        conditions_out[cond] = m
        print(f"[esm_sft/eval]   {cond}: recall@10={m['recall_at_10']:.4f} mrr={m['mrr']:.4f} hit@1={m['hit_at_1']:.4f}  ({elapsed:.1f}s)", flush=True)

    write_results(
        baseline=baseline_name, task=args.task,
        conditions=conditions_out, wandb_run_id=args.wandb_run_id,
        training_data=training_data, out_root=Path(args.results_root),
    )
    print(f"[esm_sft/eval] wrote results JSON for {baseline_name}/{args.task}", flush=True)


if __name__ == "__main__":
    main()
