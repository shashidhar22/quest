"""CPT Stage 4.2 — downstream SFT judge (plans/stage4-2.md).

Self-contained so Track A's own harness stays reproducible. It fixes three defects verified in
Track A's ESM SFT on 2026-10-05:

  1. **No MHC reached the model.** Track A's loader (`lstm_mlp/convert_data.py`) reads only
     trb_cdr3/peptide/allele-name columns, while `assemble_sequence` takes MHC from
     `mhc_one_pocket_contact`; the missing column became "". 2,800 of 2,800 sampled rows ended
     in an empty MHC slot. Here every row is serialized by the CPT trainer's own
     `project_canonical_row_to_run_input`, so the fine-tuning input is byte-for-byte the format
     each adapter was pretrained on (segments, `<eos>` separators, the row's own order_key).
  2. **Unequal trainable capacity.** Track A's base arm trained pool+head only, while the CPT
     arm also trained the adapter's LoRA. Here the base arm gets a fresh LoRA of the same shape
     (r=32, alpha=64, same target modules), so base vs CPT differs only in the pretrained init.
  3. **Eval = first N rows, full 3.5k pool.** Here: a seeded sample of up to `max_queries`
     queries, each scored against its true peptide plus 99 distractors drawn from the class-I
     candidate pool with a seed derived from the query's row_id — identical distractors for
     every arm, so arms are compared PAIRED (per-query rank differences, paired bootstrap).

Usage (on the training box, repo root = /home/ubuntu/quest):
    python -m track_a.esm_sft.judge train --task as_paired_i --adapter none \
        --tcr_variant cdr3 --mhc_variant full --out_dir RUN_DIR [--smoke]
    python -m track_a.esm_sft.judge eval  --task as_paired_i --adapter none \
        --tcr_variant cdr3 --mhc_variant full --ckpt RUN_DIR/best.pt --out RUN_DIR/eval.json [--smoke]
    python -m track_a.esm_sft.judge compare --a base/eval.json --b D/eval.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

QUEST = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QUEST / "scripts" / "training"))
from cpt_training_lib import (  # noqa: E402  (the CPT trainer's own serializer + tokenizer)
    _preprocess_for_eos_strategy,
    get_tokenizer,
    project_canonical_row_to_run_input,
)
from track_a.esm_sft.model import ESMScorer  # noqa: E402
from track_a.esm_sft_cubes.cpt_loader import load_esmc_from_local_lora  # noqa: E402
from track_a.shared.load_splits import load_as_class_i, load_candidate_pool  # noqa: E402
from track_a.shared.negative_sampling import attach_labels, sample_negatives  # noqa: E402

EOS_STRATEGY = "raw"           # what every Stage 4.1 arm trained with
POOL_SIZE = 100                # true peptide + 99 distractors (Track A's normalized-recall pool)
TRAINABLE_KEYS = ("pool.", "head.", "lora_")


# ---------------------------------------------------------------- serialization
def _clean(row: dict) -> dict:
    """pandas NaN is truthy; the CPT serializer tests `if v`, so NaN must become None."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def serialize(row: dict, tcr: str, mhc: str, tok, max_len: int) -> list[int] | None:
    text, _ = project_canonical_row_to_run_input(row, tcr, mhc)
    if not text:
        return None
    enc = tok(_preprocess_for_eos_strategy(text, EOS_STRATEGY), add_special_tokens=True,
              truncation=True, max_length=max_len)
    return [int(x) for x in enc["input_ids"]]


class RowsDS(Dataset):
    def __init__(self, df: pd.DataFrame, tcr: str, mhc: str, tok, max_len: int):
        self.rows = [_clean(r) for r in df.to_dict("records")]
        self.labels = df["label"].astype("float32").to_numpy()
        self.tcr, self.mhc, self.tok, self.max_len = tcr, mhc, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids = serialize(self.rows[i], self.tcr, self.mhc, self.tok, self.max_len) or [self.tok.cls_token_id]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[i])


def pad(seqs: list[torch.Tensor], pad_id: int):
    n, L = len(seqs), max(len(s) for s in seqs)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    am = torch.zeros((n, L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        am[i, : len(s)] = 1
    return ids, am


def collate(pad_id):
    def f(batch):
        ids, am = pad([b[0] for b in batch], pad_id)
        return ids, am, torch.stack([b[1] for b in batch])
    return f


# ---------------------------------------------------------------- model
def build_model(adapter: str, device) -> tuple[ESMScorer, object]:
    """adapter = 'none' -> fresh LoRA of the CPT shape (fair base arm); else a local adapter dir."""
    loaded = load_esmc_from_local_lora(None if adapter in ("none", "base") else adapter)
    model = ESMScorer(loaded, head_dim=256, dropout=0.1, freeze_base=True).to(device)
    return model, get_tokenizer("esmc_300m")


def n_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- data
def load_split(task: str, split: str) -> pd.DataFrame:
    df = load_as_class_i(task, split)  # ALL columns: MHC sequences, order_key, CDR1/2/3
    return df[df["peptide"].notna() & df["trb_cdr3"].notna()].reset_index(drop=True)


def with_negatives(pos: pd.DataFrame, seed: int) -> pd.DataFrame:
    # Track A / Track B parity: 5x peptide-swap negatives; (tcr, mhc) kept, so the negative
    # inherits the positive's MHC sequence columns.
    negs = sample_negatives(pos, tcr_col="trb_cdr3", pep_col="peptide", mhc_col="mhc_one_allele", seed=seed)
    return attach_labels(pos, negs).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------- train
def cmd_train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    max_pos, max_val = (200, 50) if a.smoke else (a.max_train_pos, a.max_val_pos)
    tr = load_split(a.task, "train"); va = load_split(a.task, "val")
    tr = tr.sample(n=min(max_pos, len(tr)), random_state=a.seed) if len(tr) > max_pos else tr
    va = va.sample(n=min(max_val, len(va)), random_state=a.seed) if len(va) > max_val else va
    tr, va = with_negatives(tr.reset_index(drop=True), 43), with_negatives(va.reset_index(drop=True), 44)
    model, tok = build_model(a.adapter, dev)
    pid = int(tok.pad_token_id)
    mk = lambda df, sh, bs: DataLoader(RowsDS(df, a.tcr_variant, a.mhc_variant, tok, a.max_len), batch_size=bs,
                                       shuffle=sh, num_workers=a.num_workers, collate_fn=collate(pid))
    tl, vl = mk(tr, True, a.batch_size), mk(va, False, a.batch_size * 2)
    print(f"[judge] train rows {len(tr):,} val rows {len(va):,} trainable params {n_trainable(model):,}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    best, bad, hist = -1.0, 0, []
    for ep in range(1, (1 if a.smoke else a.epochs) + 1):
        model.train(); t0 = time.time(); s = n = 0
        for i, (ids, am, y) in enumerate(tl):
            ids, am, y = ids.to(dev), am.to(dev), y.to(dev)
            loss = bce(model(ids, am), y)
            opt.zero_grad(); loss.backward(); opt.step()
            s += loss.item() * len(y); n += len(y)
            if i % 500 == 0:
                print(f"[judge]  ep{ep} step {i}/{len(tl)} loss {s / max(n, 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for ids, am, y in vl:
                P += torch.sigmoid(model(ids.to(dev), am.to(dev))).float().cpu().tolist(); Y += y.tolist()
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(Y, P)) if len(set(Y)) > 1 else 0.5
        hist.append({"epoch": ep, "train_loss": s / max(n, 1), "val_auc": auc, "secs": round(time.time() - t0, 1)})
        print(f"[judge] epoch {ep} train_loss {s / max(n, 1):.4f} val_auc {auc:.4f}", flush=True)
        if auc > best + 1e-4:
            best, bad = auc, 0
            torch.save({"args": vars(a), "trainable_state_dict":
                        {k: v for k, v in model.state_dict().items() if any(t in k for t in TRAINABLE_KEYS)}},
                       out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                break
    (out / "train_summary.json").write_text(json.dumps(
        {"args": vars(a), "best_val_auc": best, "history": hist, "trainable_params": n_trainable(model)}, indent=2))


# ---------------------------------------------------------------- eval
def _qseed(row_id) -> int:
    return int(hashlib.md5(str(row_id).encode()).hexdigest()[:8], 16)


def cmd_eval(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = build_model(a.adapter, dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    sd = model.state_dict(); sd.update(ck["trainable_state_dict"]); model.load_state_dict(sd); model.eval()
    pid = int(tok.pad_token_id)
    pool_df = load_candidate_pool()
    pool = pool_df["peptide"].astype(str).tolist() if "peptide" in pool_df else pool_df.iloc[:, 0].astype(str).tolist()
    pool_set = set(pool)
    result = {"args": vars(a), "pool_size": POOL_SIZE, "conditions": {}}
    for cond in a.conditions:
        t0 = time.time()
        df = load_split(a.task, cond)
        n_all = len(df)
        df = df[df["peptide"].astype(str).isin(pool_set)].reset_index(drop=True)
        cap = a.smoke_queries if a.smoke else a.max_queries
        if len(df) > cap:
            df = df.sample(n=cap, random_state=42).reset_index(drop=True)
        ranks, rids, trunc = [], [], 0
        for _, r in df.iterrows():
            row = _clean(r.to_dict()); true = str(row["peptide"])
            rng = np.random.default_rng(_qseed(row["row_id"]))
            others = [p for p in rng.choice(pool, size=POOL_SIZE + 8, replace=False) if p != true][: POOL_SIZE - 1]
            seqs = []
            for p in [true] + others:
                row["peptide"] = p
                ids = serialize(row, a.tcr_variant, a.mhc_variant, tok, a.max_len) or [tok.cls_token_id]
                trunc += len(ids) >= a.max_len
                seqs.append(torch.tensor(ids, dtype=torch.long))
            ids, am = pad(seqs, pid)
            with torch.no_grad():
                sc = model(ids.to(dev), am.to(dev)).float().cpu().numpy()
            ranks.append(int(1 + (sc[1:] >= sc[0]).sum()))  # ties count against the true peptide
            rids.append(str(row["row_id"]))
        rk = np.asarray(ranks)
        rng = np.random.default_rng(42)
        boots = [(rk[i] <= 10).mean() for i in (rng.integers(0, len(rk), len(rk)) for _ in range(1000))] if len(rk) else [0]
        m = {"n_rows_condition": n_all, "n_queries": int(len(rk)), "recall_at_10": float((rk <= 10).mean()) if len(rk) else None,
             "recall_at_10_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
             "mrr": float((1.0 / rk).mean()) if len(rk) else None, "random_recall_at_10": 10 / POOL_SIZE,
             "underpowered": bool(len(rk) < a.min_queries), "n_truncated_seqs": int(trunc),
             "secs": round(time.time() - t0, 1), "ranks": dict(zip(rids, ranks))}
        result["conditions"][cond] = m
        print(f"[judge] {cond}: n={m['n_queries']} recall@10={m['recall_at_10']} CI={m['recall_at_10_ci95']} "
              f"mrr={m['mrr']} truncated={trunc} ({m['secs']}s){' UNDERPOWERED' if m['underpowered'] else ''}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))


# ---------------------------------------------------------------- paired comparison
def cmd_compare(a):
    A, B = json.loads(Path(a.a).read_text()), json.loads(Path(a.b).read_text())
    for cond in sorted(set(A["conditions"]) & set(B["conditions"])):
        ra, rb = A["conditions"][cond]["ranks"], B["conditions"][cond]["ranks"]
        q = sorted(set(ra) & set(rb))
        d = np.array([(rb[k] <= 10) - (ra[k] <= 10) for k in q], dtype=float)
        rng = np.random.default_rng(42)
        boots = [d[i].mean() for i in (rng.integers(0, len(d), len(d)) for _ in range(2000))] if len(d) else [0]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "B > A" if lo > 0 else ("B < A" if hi < 0 else "no difference resolved")
        print(f"{cond}: paired n={len(q)}  delta recall@10 (B-A) = {d.mean() if len(d) else float('nan'):+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sp.add_parser(name)
        p.add_argument("--task", required=True, choices=["as_paired_i", "as_trb_i"])
        p.add_argument("--adapter", required=True, help="'none' = fresh LoRA (fair base), else adapter dir")
        p.add_argument("--tcr_variant", required=True)
        p.add_argument("--mhc_variant", required=True)
        p.add_argument("--max_len", type=int, default=1024)
        p.add_argument("--smoke", action="store_true")
    t = sp.choices["train"]
    t.add_argument("--out_dir", required=True)
    t.add_argument("--max_train_pos", type=int, default=50_000)
    t.add_argument("--max_val_pos", type=int, default=5_000)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--patience", type=int, default=1)
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--num_workers", type=int, default=4)
    e = sp.choices["eval"]
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--conditions", nargs="+", default=["test_level4", "test_novel_pep", "test_iid"])
    e.add_argument("--max_queries", type=int, default=2000)
    e.add_argument("--min_queries", type=int, default=500)
    e.add_argument("--smoke_queries", type=int, default=20)
    c = sp.add_parser("compare")
    c.add_argument("--a", required=True); c.add_argument("--b", required=True)
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
