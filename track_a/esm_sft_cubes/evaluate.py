"""Phase 1 evaluation: per-cell Recall@K on 8 cells × 2 candidate pools.

OPTIMIZED variant: pre-tokenizes the candidate pool ONCE per cell, then builds
each query×candidate input by tensor concatenation (no Python tokenization in
the hot loop). Supports multi-GPU sharding via --shard_idx/--num_shards.

Per cell: encode the full ~3,563-candidate pool ONCE → derive both fullpool and
N=100-normalized metrics from the same (n_test × n_cand) score matrix.

Output (per cell, per shard):
  <out_dir>/cell_<i>__fullpool__metrics.json   (or _shard<S>_of_<T>.json)
  <out_dir>/cell_<i>__n100__metrics.json
  <out_dir>/cell_<i>__eval_done

Usage:
    python -m track_a.esm_sft_cubes.evaluate --config configs/A_cubeA.yaml \
        [--smoke] [--cells cell_1,cell_2] [--shard_idx 0 --num_shards 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

_REPO_ROOT = "/home/ubuntu/quest"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from track_a.esm_sft.model import ESMScorer
from track_a.esm_sft_cubes.cpt_loader import load_esmc_from_local_lora
from track_a.esm_sft_cubes.cube_data import cube_chains, load_cell_test, load_full_candidate_pool
from track_a.esm_sft_cubes.dataset import TCR_COLS, MHC_COL, _safe

from quest.metrics.retrieval_metrics import compute_retrieval_metrics_from_scores
from scripts.normalized_recall import normalized_recall_k


ALL_CELLS = [f"cell_{i}" for i in range(1, 9)]


def _build_prefix_suffix(
    row: pd.Series, tcr_variant: str, mhc_variant: str, chains: str, tokenizer,
) -> tuple[list[int] | None, list[int] | None]:
    """Build prefix=[cls, TCR_tokens, eos] and suffix=[eos, MHC_tokens, eos] for a row.

    Returns (None, None) if any required column is empty.
    """
    cls = int(tokenizer.cls_token_id); eos = int(tokenizer.eos_token_id)
    tcr_cols = TCR_COLS[tcr_variant][chains]
    tcr_strs = [_safe(row.get(c)) for c in tcr_cols]
    if any(s == "" for s in tcr_strs):
        return None, None
    mhc_str = _safe(row.get(MHC_COL[mhc_variant]))
    if not mhc_str:
        return None, None
    # Prefix: cls + tcr_1 + eos + tcr_2 + eos + ... + tcr_n + eos
    prefix = [cls]
    for s in tcr_strs:
        prefix.extend(tokenizer.encode(s, add_special_tokens=False))
        prefix.append(eos)
    # Suffix: [<-- peptide goes here] + eos + MHC + eos
    suffix = [eos] + list(tokenizer.encode(mhc_str, add_special_tokens=False)) + [eos]
    return prefix, suffix


@torch.no_grad()
def score_test_cell_fast(
    model: ESMScorer, loaded, test_df: pd.DataFrame, candidates: list[str],
    tcr_variant: str, mhc_variant: str, chains: str,
    max_len: int, batch_size: int, device: torch.device,
) -> np.ndarray:
    """Optimized scoring: pre-tokenize candidates once, batch-build inputs.

    Returns (n_test, n_cand) float32 score matrix.
    """
    tok = loaded.tokenizer
    pad_id = getattr(tok, "pad_token_id", None) or 1
    n_test = len(test_df); n_cand = len(candidates)

    # Pre-tokenize all candidates ONCE
    cand_ids: list[list[int]] = [list(tok.encode(c, add_special_tokens=False)) for c in candidates]
    cand_lens = np.array([len(c) for c in cand_ids], dtype=np.int32)

    scores = np.zeros((n_test, n_cand), dtype=np.float32)

    for q_i in range(n_test):
        row = test_df.iloc[q_i]
        prefix, suffix = _build_prefix_suffix(row, tcr_variant, mhc_variant, chains, tok)
        if prefix is None:
            scores[q_i, :] = -np.inf; continue
        Lp = len(prefix); Ls = len(suffix)
        prefix_t = torch.as_tensor(prefix, dtype=torch.long)
        suffix_t = torch.as_tensor(suffix, dtype=torch.long)

        # Iterate candidates in batches, building (B, L) tensors directly
        for batch_start in range(0, n_cand, batch_size):
            batch_end = min(batch_start + batch_size, n_cand)
            B = batch_end - batch_start
            # Length per row, truncated to max_len
            row_lens = np.minimum(Lp + cand_lens[batch_start: batch_end] + Ls, max_len)
            max_l = int(row_lens.max())
            ids = torch.full((B, max_l), pad_id, dtype=torch.long)
            attn = torch.zeros((B, max_l), dtype=torch.long)
            for k, j in enumerate(range(batch_start, batch_end)):
                cand_t = torch.as_tensor(cand_ids[j], dtype=torch.long)
                row_seq = torch.cat([prefix_t, cand_t, suffix_t])[: max_len]
                L = row_seq.size(0)
                ids[k, : L] = row_seq
                attn[k, : L] = 1
            ids = ids.to(device, non_blocking=True); attn = attn.to(device, non_blocking=True)
            logits = model(ids, attn).float().cpu().numpy()  # (B,)
            scores[q_i, batch_start: batch_end] = logits
    return scores


def _load_trained_model(model_dir: Path, cfg: dict, device: torch.device):
    loaded = load_esmc_from_local_lora(cfg["ckpt_path"])
    model = ESMScorer(loaded, head_dim=256, dropout=0.1, freeze_base=True).to(device)
    state = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
    sd = state["trainable_state_dict"]
    msg = model.load_state_dict(sd, strict=False)
    if msg.unexpected_keys:
        print(f"[eval] WARN unexpected_keys: {msg.unexpected_keys[:5]} ...")
    model.eval()
    return model, loaded


def _compute_metrics_and_write(
    scores: np.ndarray, candidates: list[str], test_df: pd.DataFrame,
    cfg: dict, cell: str, out_dir: Path,
) -> None:
    cube_name = cfg["cube_name"]; n_test, n_cand = scores.shape
    cand_to_idx = {p: i for i, p in enumerate(candidates)}

    # Full pool
    true_idx = np.array([cand_to_idx[str(p).upper()] for p in test_df["peptide"]])
    m_full = compute_retrieval_metrics_from_scores(scores, true_idx)
    m_full = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in m_full.items()}
    m_full.update({"n_test": int(n_test), "pool": "full", "pool_size": int(n_cand),
                   "ckpt_name": cfg["ckpt_name"], "cube_name": cube_name, "cell": cell})
    (out_dir / f"{cell}__fullpool__metrics.json").write_text(json.dumps(m_full, indent=2))

    # N=100 normalized
    true_peps = test_df["peptide"].astype(str).str.upper().values
    is_gold = np.tile(candidates, n_test) == np.repeat(true_peps, n_cand)
    long_df = pd.DataFrame({
        "query_idx": np.repeat(np.arange(n_test), n_cand),
        "candidate_pep": np.tile(candidates, n_test),
        "score": scores.reshape(-1).astype(np.float32),
        "is_gold": is_gold,
    })
    try:
        m_n100 = normalized_recall_k(long_df, k_values=(1, 5, 10, 20),
                                      pool_size=int(cfg.get("n100_pool_size", 100)),
                                      n_seeds=int(cfg.get("n100_seeds", 10)),
                                      base_seed=int(cfg.get("n100_base_seed", 42)))
    except Exception as e:
        m_n100 = {"error": f"{type(e).__name__}: {e}"}
    m_n100.update({"n_test": int(n_test), "pool": "n100",
                   "ckpt_name": cfg["ckpt_name"], "cube_name": cube_name, "cell": cell})
    (out_dir / f"{cell}__n100__metrics.json").write_text(json.dumps(m_n100, indent=2))


def evaluate_cell(
    cell: str, cfg: dict, out_dir: Path, model, loaded, candidates: list[str],
    device: torch.device, batch_size: int, smoke: bool,
    shard_idx: int = 0, num_shards: int = 1,
) -> None:
    marker = out_dir / f"{cell}__eval_done"
    if marker.exists() and not smoke and num_shards == 1:
        print(f"[eval] {cell}: done, skip"); return
    cube_name = cfg["cube_name"]
    cand_to_idx = {p: i for i, p in enumerate(candidates)}

    max_rows = 50 if smoke else int(cfg.get("max_test_rows", 2000))
    test_df = load_cell_test(cube_name, cell, cfg["tcr_variant"], cfg["mhc_variant"], max_test_rows=max_rows)
    test_df = test_df[test_df["peptide"].astype(str).str.upper().isin(cand_to_idx)].reset_index(drop=True)
    n_test = len(test_df)

    # Apply shard
    if num_shards > 1:
        idxs = np.arange(n_test)
        my_idxs = idxs[shard_idx :: num_shards]
        test_df = test_df.iloc[my_idxs].reset_index(drop=True)
        n_test = len(test_df)
        print(f"[eval] {cell}: shard {shard_idx}/{num_shards}  n_test={n_test}", flush=True)
    else:
        print(f"[eval] {cell}: n_test={n_test}", flush=True)

    if n_test == 0:
        if num_shards == 1:
            for tag in ["fullpool", "n100"]:
                (out_dir / f"{cell}__{tag}__metrics.json").write_text(json.dumps({
                    "n_test": 0, "pool": tag, "note": "no rows with peptide in pool"}, indent=2))
            marker.write_text("empty")
        return

    chains = cube_chains(cube_name)
    t0 = time.time()
    scores = score_test_cell_fast(
        model, loaded, test_df, candidates,
        cfg["tcr_variant"], cfg["mhc_variant"], chains,
        max_len=int(cfg.get("max_seq_len", 320)),
        batch_size=batch_size, device=device,
    )
    elapsed = time.time() - t0
    print(f"[eval] {cell}: scored {n_test}×{len(candidates)} in {elapsed:.1f}s "
          f"({elapsed/max(n_test,1):.2f}s/query)", flush=True)

    if num_shards > 1:
        # Save partial scores; aggregation happens elsewhere
        shard_path = out_dir / "scores_partial" / f"{cell}__shard{shard_idx}of{num_shards}.npz"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(shard_path, scores=scores, peptides=test_df["peptide"].astype(str).str.upper().values)
        return

    _compute_metrics_and_write(scores, candidates, test_df, cfg, cell, out_dir)
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ"))


def aggregate_shards(cell: str, cfg: dict, out_dir: Path, candidates: list[str], num_shards: int) -> None:
    """Combine shard score files into the full (n_test, n_cand) matrix and emit metrics."""
    cube_name = cfg["cube_name"]
    shard_files = sorted((out_dir / "scores_partial").glob(f"{cell}__shard*of{num_shards}.npz"))
    if len(shard_files) != num_shards:
        raise FileNotFoundError(f"expected {num_shards} shard files for {cell}, got {len(shard_files)}")
    score_chunks, pep_chunks = [], []
    for f in shard_files:
        d = np.load(f, allow_pickle=True)
        score_chunks.append(d["scores"]); pep_chunks.append(d["peptides"])
    scores = np.concatenate(score_chunks, axis=0)
    peps = np.concatenate(pep_chunks, axis=0)
    df = pd.DataFrame({"peptide": peps})
    _compute_metrics_and_write(scores, candidates, df, cfg, cell, out_dir)
    (out_dir / f"{cell}__eval_done").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cells", default=None)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--aggregate_only", action="store_true",
                   help="Skip scoring; just aggregate existing shard files into metrics")
    args = p.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    ckpt_name = cfg["ckpt_name"]; cube_name = cfg["cube_name"]
    out_dir = Path(cfg["output_dir"]) / f"{ckpt_name}__{cube_name}"
    if not (out_dir / "best.pt").exists():
        raise FileNotFoundError(f"best.pt not found in {out_dir} — train first")

    candidates = load_full_candidate_pool()
    cells = args.cells.split(",") if args.cells else ALL_CELLS

    if args.aggregate_only:
        for cell in cells:
            try:
                aggregate_shards(cell, cfg, out_dir, candidates, args.num_shards)
                print(f"[eval] {cell}: aggregated {args.num_shards} shards", flush=True)
            except Exception as e:
                print(f"[eval] {cell}: aggregation failed: {e}")
        done = all((out_dir / f"{c}__eval_done").exists() for c in ALL_CELLS)
        if done:
            (out_dir / "_eval_done").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ"))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loaded = _load_trained_model(out_dir, cfg, device)
    print(f"[eval] candidate pool: {len(candidates)} peptides", flush=True)
    batch_size = int(cfg.get("eval_batch_size", 64))

    for cell in cells:
        evaluate_cell(cell, cfg, out_dir, model, loaded, candidates, device, batch_size, args.smoke,
                      shard_idx=args.shard_idx, num_shards=args.num_shards)

    if args.num_shards == 1:
        done = all((out_dir / f"{c}__eval_done").exists() for c in ALL_CELLS)
        if done and not args.smoke:
            (out_dir / "_eval_done").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ"))
            print("[eval] all 8 cells done", flush=True)


if __name__ == "__main__":
    main()
