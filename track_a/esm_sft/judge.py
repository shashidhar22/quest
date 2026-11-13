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
    annotate_token_regions,
    get_tokenizer,
    project_canonical_row_to_run_input,
)
from track_a.esm_sft.model import ESMScorer  # noqa: E402
from track_a.esm_sft.scoring import (  # noqa: E402  (NumPy only; see its docstring)
    SFT_OBJECTIVE,
    SFT_TRAIN_DEFAULTS as D,
    DeviceError,
    ScoreError,
    cotenants,
    incompatible_checkpoint_args,
    missing_trainable_keys,
    rank_of,
    score_diagnostics,
    select_device,
    sft_recipe,
)
from track_a.shared import artifacts as A  # noqa: E402  (stdlib only; digests and lineage)
from track_a.esm_sft_cubes.cpt_loader import load_esmc_from_local_lora  # noqa: E402
from track_a.shared.load_splits import load_as_class_i, load_candidate_pool  # noqa: E402
from track_a.shared.negative_sampling import attach_labels, sample_negatives  # noqa: E402

EOS_STRATEGY = "raw"           # what every Stage 4.1 arm trained with
POOL_SIZE = 100                # true peptide + 99 distractors (Track A's normalized-recall pool)
# `pool_peptide.` is listed separately: "pool." does not match "pool_peptide.score.weight", so
# without it the peptide pool would be dropped from every checkpoint and silently reloaded at its
# random initialisation — the defect audit finding 2 (2026-11-06) closed for the other tensors.
TRAINABLE_KEYS = ("pool.", "pool_peptide.", "head.", "lora_")


# ---------------------------------------------------------------- serialization
def _clean(row: dict) -> dict:
    """pandas NaN is truthy; the CPT serializer tests `if v`, so NaN must become None."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def _special_ids(tok) -> set:
    return {int(i) for i in (tok.cls_token_id, tok.eos_token_id, tok.pad_token_id) if i is not None}


def serialize(row: dict, tcr: str, mhc: str, tok, max_len: int):
    """``(token ids, peptide mask)`` for one row, or None when it serializes to nothing.

    The peptide mask is what `ESMScorer` pools the peptide block over, and it is derived from
    the SAME text and segment kinds the ids came from, so truncation is accounted for: a
    peptide pushed past ``max_len`` leaves an all-zero mask and the model refuses the row
    rather than scoring a molecule whose peptide never reached it.
    """
    text, kinds = project_canonical_row_to_run_input(row, tcr, mhc)
    if not text:
        return None
    enc = tok(_preprocess_for_eos_strategy(text, EOS_STRATEGY), add_special_tokens=True,
              truncation=True, max_length=max_len)
    ids = [int(x) for x in enc["input_ids"]]
    regions = annotate_token_regions(text, row, kinds, ids,
                                     separator_token_id=int(tok.eos_token_id),
                                     special_token_ids=_special_ids(tok))
    return ids, [1 if r == "peptide" else 0 for r in regions]


def serialized_or_refuse(row: dict, tcr: str, mhc: str, tok, max_len: int, where: str):
    """`serialize`, with the silent `or [cls]` fallback removed.

    That fallback turned a row the serializer could make nothing of into a one-token molecule
    and scored it anyway. A judge that cannot see the peptide is the instrument finding 9.14
    describes, so this refuses and names the row instead.
    """
    out = serialize(row, tcr, mhc, tok, max_len)
    if out is None or not any(out[1]):
        raise ScoreError(
            f"{where}: row_id={row.get('row_id')!r} serialized to "
            f"{'nothing' if out is None else 'no peptide token'} under tcr={tcr} mhc={mhc} "
            f"max_len={max_len} (order_key={row.get('order_key')!r}). Scoring it would feed the "
            f"model a molecule with no peptide in it; fix the row or the serialization.")
    return out


class RowsDS(Dataset):
    def __init__(self, df: pd.DataFrame, tcr: str, mhc: str, tok, max_len: int):
        self.rows = [_clean(r) for r in df.to_dict("records")]
        self.labels = df["label"].astype("float32").to_numpy()
        self.tcr, self.mhc, self.tok, self.max_len = tcr, mhc, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, pep = serialized_or_refuse(self.rows[i], self.tcr, self.mhc, self.tok,
                                        self.max_len, "training row")
        return ids, pep, torch.tensor(self.labels[i])


def pad(rows, pad_id: int):
    """``[(ids, peptide mask), ...]`` -> ``(ids, attention mask, peptide mask)``."""
    n, L = len(rows), max(len(r[0]) for r in rows)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    am = torch.zeros((n, L), dtype=torch.long)
    pep = torch.zeros((n, L), dtype=torch.long)
    for i, (s, p) in enumerate(rows):
        ids[i, : len(s)] = torch.as_tensor(s, dtype=torch.long)
        am[i, : len(s)] = 1
        pep[i, : len(p)] = torch.as_tensor(p, dtype=torch.long)
    return ids, am, pep


def collate(pad_id):
    def f(batch):
        ids, am, pep = pad([(b[0], b[1]) for b in batch], pad_id)
        return ids, am, pep, torch.stack([b[2] for b in batch])
    return f


# ---------------------------------------------------------------- model
def build_model(adapter: str, device) -> tuple[ESMScorer, object]:
    """adapter = 'none' -> fresh LoRA of the CPT shape (fair base arm); else a local adapter dir."""
    loaded = load_esmc_from_local_lora(None if adapter in ("none", "base") else adapter)
    model = ESMScorer(loaded, head_dim=256, dropout=0.1, freeze_base=True, peptide_pool=True).to(device)
    return model, get_tokenizer("esmc_300m")


def load_checkpoint(model, path, device, want=None):
    """Load a trained checkpoint into `model`, refusing an incomplete one.

    Audit finding 2, 2026-11-06: this used to be `sd.update(saved); load_state_dict(sd)` over a
    fresh `model.state_dict()`, so a trainable tensor absent from the file kept its random
    initialisation and the load still reported success. An arm evaluated that way is not the arm
    it claims to be, and nothing downstream could tell. The expected set comes from the live
    model, so it cannot drift from the model's own structure.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    if "trainable_state_dict" not in ck:
        raise ScoreError(f"{path} has no 'trainable_state_dict'; this is not a judge checkpoint")
    saved = ck["trainable_state_dict"]
    live = model.state_dict()
    missing, unexpected = missing_trainable_keys(saved.keys(), live.keys(), TRAINABLE_KEYS)
    if missing or unexpected:
        raise ScoreError(
            f"{path} does not match this model's trainable parameters, so evaluating it would "
            f"score partly untrained weights as if trained.\n"
            f"  missing from the checkpoint ({len(missing)}): {missing[:8]}\n"
            f"  not present in the model ({len(unexpected)}): {unexpected[:8]}")
    if want is not None:
        clash = incompatible_checkpoint_args(ck.get("args"), want)
        if clash:
            raise ScoreError(
                f"{path} was not trained for this evaluation:\n  " + "\n  ".join(clash))
    live.update(saved)
    model.load_state_dict(live)
    model.eval()
    return ck


def n_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- data
def load_split(task: str, split: str) -> pd.DataFrame:
    """One row per MOLECULE. The benchmark splits are exploded across segment orderings
    (`order_key`): as_paired_i test_iid is 6,696 rows for 201 `row_id`s, up to 120 orderings
    each; train is 553,440 rows for 15,145. Scoring every ordering inflated n 25x, made the
    bootstrap CI ~5x too narrow, and cost 10 GPU-hours per cell (2026-10-07). Keep the first
    ordering of each `row_id`; `row_id` is also the paired-comparison key, so it must be unique."""
    df = load_as_class_i(task, split)  # ALL columns: MHC sequences, order_key, CDR1/2/3
    df = df[df["peptide"].notna() & df["trb_cdr3"].notna()]
    return df.drop_duplicates(subset="row_id", keep="first").reset_index(drop=True)


def with_negatives(pos: pd.DataFrame, seed: int) -> pd.DataFrame:
    # Track A / Track B parity: 5x peptide-swap negatives; (tcr, mhc) kept, so the negative
    # inherits the positive's MHC sequence columns.
    negs = sample_negatives(pos, tcr_col="trb_cdr3", pep_col="peptide", mhc_col="mhc_one_allele", seed=seed)
    return attach_labels(pos, negs).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------- train
def cmd_train(a):
    if a.batch_size < 2:
        raise ScoreError("batch_size must be at least 2 for the normalized judge head")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device(select_device(torch.cuda.is_available(),
                                    allow_cpu=getattr(a, "allow_cpu", False)))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    max_pos, max_val = (200, 50) if a.smoke else (a.max_train_pos, a.max_val_pos)
    tr = load_split(a.task, "train"); va = load_split(a.task, "val")
    tr = tr.sample(n=min(max_pos, len(tr)), random_state=a.seed) if len(tr) > max_pos else tr
    va = va.sample(n=min(max_val, len(va)), random_state=a.seed) if len(va) > max_val else va
    tr, va = with_negatives(tr.reset_index(drop=True), 43), with_negatives(va.reset_index(drop=True), 44)
    if len(tr) < a.batch_size:
        raise ScoreError(
            f"training has {len(tr)} rows, fewer than batch_size={a.batch_size}; "
            "drop_last would train zero batches. Reduce batch_size or supply more molecules.")
    model, tok = build_model(a.adapter, dev)
    pid = int(tok.pad_token_id)
    # drop_last on the TRAINING loader only: `build_head` starts with BatchNorm1d, which raises
    # on a batch of one in train mode ("Expected more than 1 value per channel"), and a trailing
    # singleton is whatever len(rows) % batch_size happens to be. Evaluation runs in eval() on
    # running statistics, so its last batch is kept and every row is scored.
    mk = lambda df, sh, bs: DataLoader(RowsDS(df, a.tcr_variant, a.mhc_variant, tok, a.max_len), batch_size=bs,
                                       shuffle=sh, num_workers=a.num_workers, collate_fn=collate(pid),
                                       drop_last=sh)
    tl, vl = mk(tr, True, a.batch_size), mk(va, False, a.batch_size * 2)
    print(f"[judge] train rows {len(tr):,} (molecules x 1+5 negatives) val rows {len(va):,} "
          f"trainable params {n_trainable(model):,}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    best, bad, hist = -1.0, 0, []
    for ep in range(1, (1 if a.smoke else a.epochs) + 1):
        model.train(); t0 = time.time(); s = n = 0
        for i, (ids, am, pep, y) in enumerate(tl):
            ids, am, pep, y = ids.to(dev), am.to(dev), pep.to(dev), y.to(dev)
            loss = bce(model(ids, am, pep), y)
            opt.zero_grad(); loss.backward(); opt.step()
            s += loss.item() * len(y); n += len(y)
            if i % 500 == 0:
                el = max(time.time() - t0, 1e-6)
                print(f"[judge]  ep{ep} step {i}/{len(tl)} loss {s / max(n, 1):.4f} ({el:.0f}s, "
                      f"{n / el:.1f} rows/s)", flush=True)
        model.eval(); P, Y = [], []
        val_loss_sum = 0.0
        with torch.no_grad():
            for ids, am, pep, y in vl:
                logits = model(ids.to(dev), am.to(dev), pep.to(dev))
                val_loss_sum += bce(logits, y.to(dev)).item() * len(y)
                P += torch.sigmoid(logits).float().cpu().tolist()
                Y += y.tolist()
        from sklearn.metrics import roc_auc_score
        if not Y:
            raise ScoreError("validation has no rows; refusing to publish an unvalidated checkpoint")
        auc = float(roc_auc_score(Y, P)) if len(set(Y)) > 1 else 0.5
        val_loss = val_loss_sum / len(Y)
        positive_fraction = sum(Y) / len(Y)
        prior_loss = (-positive_fraction * math.log(positive_fraction)
                      - (1 - positive_fraction) * math.log(1 - positive_fraction)
                      if 0 < positive_fraction < 1 else 0.0)
        quality = {"val_loss": val_loss, "val_prior_loss": prior_loss,
                   "val_loss_beats_prior": val_loss < prior_loss}
        hist.append({"epoch": ep, "train_loss": s / max(n, 1), "val_auc": auc,
                     **quality, "secs": round(time.time() - t0, 1)})
        print(f"[judge] epoch {ep} train_loss {s / max(n, 1):.4f} val_auc {auc:.4f} "
              f"val_loss {val_loss:.4f} prior_loss {prior_loss:.4f}", flush=True)
        sd = {k: v for k, v in model.state_dict().items() if any(t in k for t in TRAINABLE_KEYS)}
        # Keep the latest epoch too: on 2026-10-07 a below-chance val AUC made selection keep the
        # untrained epoch 1, and the epoch-2 weights were lost with nothing to re-evaluate.
        # The training inputs are hashed into the checkpoint, so a later evaluation can say
        # which data produced the weights it is reading rather than which data is there now.
        meta = {"args": vars(a), "data_digests": A.split_digests(a.task, ("train", "val")),
                "sft_objective": SFT_OBJECTIVE, "cotenants": cotenants(),
                "code_commit": A.code_commit(), **quality}
        torch.save({**meta, "epoch": ep, "val_auc": auc, "trainable_state_dict": sd}, out / "last.pt")
        if auc > best + 1e-4:
            best, bad = auc, 0
            torch.save({**meta, "val_auc": auc, "trainable_state_dict": sd}, out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                break
    (out / "train_summary.json").write_text(json.dumps(
        {"args": vars(a), "best_val_auc": best, "history": hist, "trainable_params": n_trainable(model)}, indent=2))


# ---------------------------------------------------------------- eval
def cluster_bootstrap_ci(hits, peptides, n_boot: int = 2000, seed: int = 42, pct=(2.5, 97.5)):
    """95 % CI for a mean, resampling PEPTIDES rather than queries.

    Queries that share a true peptide are not independent draws for a peptide-retrieval metric:
    `as_paired_i` test_iid holds 201 molecules over 53 peptides, 93 of them on three peptides. A
    per-query bootstrap treated them as 201 independent trials and came out 3-4x too narrow,
    which manufactured a "significantly below random" reading of test_novel_pep's 0.081
    (2026-10-15, quest-cpt/findings.md 9.6). This is 9.4's rows-vs-molecules error one level up.

    Returns ``(lo, hi, n_clusters)``.
    """
    hits = np.asarray(hits, dtype=float)
    peptides = np.asarray(peptides)
    if not len(hits):
        return (0.0, 0.0, 0)
    groups = [np.flatnonzero(peptides == v) for v in pd.unique(peptides)]
    k = len(groups)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, k, k)])
        boots[b] = hits[idx].mean()
    lo, hi = np.percentile(boots, pct)
    return (float(lo), float(hi), k)


def _qseed(row_id) -> int:
    return int(hashlib.md5(str(row_id).encode()).hexdigest()[:8], 16)


def _pool_by_length(pool) -> dict:
    """Pool peptides indexed by length, for the length-matched distractor draw."""
    by_len: dict = {}
    for q in pool:
        by_len.setdefault(len(q), []).append(q)
    return {k: np.asarray(v, dtype=object) for k, v in by_len.items()}


def _draw_distractors(pool, by_len, true, rng, n, mode):
    """``n`` distractors for one query, and how many share the true peptide's length.

    ``uniform`` is the draw wave 1 shipped on 2026-10-08. It confounds the metric: the
    pool is 76 % 9-mers while the true peptide's length varies, so inside as_paired_i
    test_level4 9-mers scored 0.553 and 14-mers 0.013 — an order of magnitude on peptide
    length alone (quest-cpt/findings.md 9.6). A distractor five residues shorter than the
    true peptide is separable with no receptor information at all, so a uniform draw
    measures length as much as specificity.

    ``length_matched`` draws only same-length peptides, falling back to the nearest
    lengths when the pool holds too few. The shortfall is counted and reported per
    condition rather than hidden: a query padded from other lengths stays partly
    confounded, and the reader needs to know which queries those are.

    Returns ``(others, n_same_length)``.
    """
    if mode == "uniform":
        # Preserve wave 1's exact RNG call sequence, so --distractor_draw uniform
        # reproduces the 2026-10-08 numbers bit for bit.
        others = [q for q in rng.choice(pool, size=n + 9, replace=False) if q != true][:n]
        return others, sum(1 for q in others if len(q) == len(true))

    L = len(true)
    same = [q for q in by_len.get(L, ()) if q != true]
    if len(same) >= n:
        picked = rng.choice(np.asarray(same, dtype=object), size=n, replace=False)
        return list(picked), n

    out = list(rng.permutation(np.asarray(same, dtype=object))) if same else []
    n_same = len(out)
    limit = max(by_len) if by_len else 0
    for d in range(1, limit + 1):
        if len(out) >= n:
            break
        near = [q for L2 in (L - d, L + d) for q in by_len.get(L2, ()) if q != true]
        if not near:
            continue
        take = min(n - len(out), len(near))
        out += list(rng.choice(np.asarray(near, dtype=object), size=take, replace=False))
    return out[:n], n_same


def cmd_eval(a):
    dev = torch.device(select_device(torch.cuda.is_available(),
                                    allow_cpu=getattr(a, "allow_cpu", False)))
    model, tok = build_model(a.adapter, dev)
    # The checkpoint is KEPT: its recorded training args are the only honest source for this
    # evaluation's SFT lineage. Discarding it and reading `getattr(a, "seed", None)` off the
    # eval's own namespace recorded a structural null, because --seed exists only on the train
    # parser (round two of the 2026-11-06 review, finding 4).
    ck = load_checkpoint(model, a.ckpt, dev, want=vars(a))
    ck_args = ck.get("args") or {}
    pid = int(tok.pad_token_id)
    pool_df = load_candidate_pool()
    pool = pool_df["peptide"].astype(str).tolist() if "peptide" in pool_df else pool_df.iloc[:, 0].astype(str).tolist()
    pool_set = set(pool)
    by_len = _pool_by_length(pool)
    # Artifact-derived provenance (audit finding 1, 2026-11-06). The launcher used to DECLARE
    # which pool a run used while this function resolved its own; a manifest is now derived from
    # this block, so it describes the files that produced the numbers beside it.
    result = {"args": vars(a), "pool_size": POOL_SIZE, "conditions": {},
              "provenance": {
                  "splits_dir": str(A.splits_dir()),
                  "pool_path": str(A.pool_path()),
                  "pool_digest": A.file_digest(A.pool_path()),
                  "split_digests": A.split_digests(a.task, a.conditions),
                  "ckpt_path": str(a.ckpt),
                  "ckpt_digest": A.file_digest(a.ckpt),
                  "device": str(dev),
                  "torch_version": torch.__version__,
                  "code_commit": A.code_commit(),
                  "pool_size": POOL_SIZE,
                  # The SFT side, from the checkpoint's own record of its training
                  "sft_objective": SFT_OBJECTIVE,
                  "sft_seed": ck_args.get("seed"),
                  "sft_max_len": ck_args.get("max_len"),
                  "sft_recipe": sft_recipe(ck_args),
                  "sft_train_digests": ck.get("data_digests"),
                  "cotenants": cotenants(),
                  **A.cpt_lineage(a.adapter),
              }}
    for cond in a.conditions:
        t0 = time.time()
        df = load_split(a.task, cond)
        n_all = len(df)
        df = df[df["peptide"].astype(str).isin(pool_set)].reset_index(drop=True)
        cap = a.smoke_queries if a.smoke else a.max_queries
        if len(df) > cap:
            df = df.sample(n=cap, random_state=42).reset_index(drop=True)
        ranks, rids, true_peps, same_len, trunc = [], [], [], [], 0
        diag = {"spread_min": None, "n_all_tied": 0, "n_distinct_min": None}
        for _, r in df.iterrows():
            row = _clean(r.to_dict()); true = str(row["peptide"])
            true_peps.append(true)
            rng = np.random.default_rng(_qseed(row["row_id"]))
            others, n_same = _draw_distractors(
                pool, by_len, true, rng, POOL_SIZE - 1, a.distractor_draw)
            same_len.append(n_same)
            seqs = []
            for p in [true] + others:
                row["peptide"] = p
                one = serialized_or_refuse(row, a.tcr_variant, a.mhc_variant, tok, a.max_len,
                                           f"{cond} candidate {p!r}")
                trunc += len(one[0]) >= a.max_len
                seqs.append(one)
            ids, am, pep = pad(seqs, pid)
            with torch.no_grad():
                sc = model(ids.to(dev), am.to(dev), pep.to(dev)).float().cpu().numpy()
            # rank_of refuses a non-finite score instead of turning it into rank 1. The raise
            # propagates: an eval that cannot score every query writes NO output file, so a
            # numerical failure can never reach the gate or the comparator as a result.
            d = score_diagnostics(sc)
            try:
                ranks.append(rank_of(sc))
            except ScoreError as e:
                raise ScoreError(
                    f"{cond} row_id={row['row_id']}: {e}\n"
                    f"  diagnostics: {d}\n"
                    f"  No eval.json is written. Re-run the probe against this checkpoint "
                    f"(`judge probe --ckpt ...`) before spending anything further.") from None
            rids.append(str(row["row_id"]))
            diag["n_all_tied"] += int(d["all_tied"])
            for key, val in (("spread_min", d["spread"]), ("n_distinct_min", d["n_distinct"])):
                diag[key] = val if diag[key] is None else min(diag[key], val)
        peps = np.asarray(true_peps)
        rk = np.asarray(ranks)
        # Cluster-bootstrap over PEPTIDES, not queries. Queries sharing a true peptide are not
        # independent for a peptide-retrieval metric: as_paired_i test_iid is 201 molecules over
        # 53 peptides, 93 of them in the top three. Resampling queries made the CI 3-4x too narrow
        # and turned test_novel_pep's 0.081 into a spurious "significantly below random"
        # (2026-10-15, quest-cpt/findings.md 9.6). Same error one level up from 9.4's rows-vs-
        # molecules: molecules are not the unit of independence either.
        hits = (rk <= 10).astype(float)
        ci_lo, ci_hi, n_pep = cluster_bootstrap_ci(hits, peps)
        m = {"n_rows_condition": n_all, "n_queries": int(len(rk)),
             # the metric's real sample size: distinct true peptides, not queries
             "n_peptides": n_pep,
             "recall_at_10": float((rk <= 10).mean()) if len(rk) else None,
             "recall_at_10_ci95": [ci_lo, ci_hi],
             "mrr": float((1.0 / rk).mean()) if len(rk) else None, "random_recall_at_10": 10 / POOL_SIZE,
             # power is set by distinct peptides, not queries (see the cluster bootstrap above)
             "underpowered": bool(n_pep < a.min_peptides), "n_truncated_seqs": int(trunc),
             # which draw produced this number: a length_matched score is NOT comparable
             # to a uniform one, and the 2026-10-08 wave-1 cell was uniform
             "distractor_draw": a.distractor_draw,
             "frac_distractors_same_length": (
                 float(np.mean(same_len) / (POOL_SIZE - 1)) if same_len else None),
             "n_queries_fully_length_matched": int(
                 sum(1 for v in same_len if v == POOL_SIZE - 1)),
             # a rank alone cannot tell a dead scorer from an unlucky one: an all-tie query
             # and a genuinely last-placed one both score rank N (wave 1, 18 GPU-hours).
             "score_spread_min": diag["spread_min"],
             "score_n_distinct_min": diag["n_distinct_min"],
             "n_queries_all_candidates_tied": diag["n_all_tied"],
             "secs": round(time.time() - t0, 1), "ranks": dict(zip(rids, ranks)),
             "true_peptides": dict(zip(rids, true_peps))}
        result["conditions"][cond] = m
        print(f"[judge] {cond}: n={m['n_queries']} molecules over {m['n_peptides']} peptides "
              f"recall@10={m['recall_at_10']} CI={m['recall_at_10_ci95']} (peptide-clustered) "
              f"mrr={m['mrr']} truncated={trunc} ({m['secs']}s)"
              f"{' UNDERPOWERED' if m['underpowered'] else ''}", flush=True)
        print(f"[judge] {cond}: min score spread={m['score_spread_min']} "
              f"min distinct candidates={m['score_n_distinct_min']} "
              f"fully tied queries={m['n_queries_all_candidates_tied']}", flush=True)
        print(f"[judge] {cond}: draw={m['distractor_draw']} "
              f"same-length distractors={m['frac_distractors_same_length']:.3f} "
              f"fully matched queries={m['n_queries_fully_length_matched']}/{m['n_queries']}",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))


# ---------------------------------------------------------------- probe (the smoke gate)
def cmd_probe(a):
    """Does the scorer's output actually depend on the input? Three real rows, each scored with
    its true peptide and 5 distractors, at full precision. Exit 1 if any logit is non-finite or
    if any query's 6 candidates come out identical — that is exactly the failure wave 1 spent
    18 GPU-hours discovering (every candidate tied, Recall@10 = 0). Also prints the spread of
    the pooled vectors, so an input-independent ENCODER shows up as its own line.

    With --ckpt it probes THAT checkpoint. Without it, it probes a freshly initialised head on
    the adapter, which is the right check before training but vouches for nothing afterwards:
    the 2026-11-06 audit noted that a probe of a fresh model says nothing about the base or
    trained checkpoint the paid evaluation then reads. An incomplete checkpoint refuses here,
    before any evaluation is bought."""
    from track_a.esm_sft.encoders import encoder_forward
    dev = torch.device(select_device(torch.cuda.is_available(),
                                    allow_cpu=getattr(a, "allow_cpu", False)))
    model, tok = build_model(a.adapter, dev); model.eval()
    if getattr(a, "ckpt", None):
        try:
            # want=vars(a): the probe is the cheap step, so an incompatible checkpoint should
            # refuse HERE rather than after a paid evaluation has loaded it.
            load_checkpoint(model, a.ckpt, dev, want=vars(a))
        except ScoreError as e:
            print(f"[probe] FAIL: {e}", flush=True)
            sys.exit(1)
        print(f"[probe] checkpoint loaded with every expected trainable key: {a.ckpt}", flush=True)
    else:
        print("[probe] no --ckpt: probing a FRESH head, which vouches for no checkpoint",
              flush=True)
    pid = int(tok.pad_token_id)
    pool = load_candidate_pool(); pool = (pool["peptide"] if "peptide" in pool else pool.iloc[:, 0]).astype(str).tolist()
    df = load_split(a.task, "test_iid").head(3)
    ok = True
    for _, r in df.iterrows():
        row = _clean(r.to_dict()); true = str(row["peptide"])
        rng = np.random.default_rng(_qseed(row["row_id"]))
        cands = [true] + [p for p in rng.choice(pool, size=12, replace=False) if p != true][:5]
        seqs = []
        for p in cands:
            row["peptide"] = p
            seqs.append(serialized_or_refuse(row, a.tcr_variant, a.mhc_variant, tok, a.max_len,
                                             f"probe candidate {p!r}"))
        ids, am, pep = pad(seqs, pid)
        with torch.no_grad():
            emb = encoder_forward(model.loaded, ids.to(dev), am.to(dev)).float()
            valid = am.to(dev).bool()
            pooled_all = model.pool(emb, valid)
            pooled_pep = model.pool_peptide(emb, pep.to(dev).bool() & valid)
            sc = model(ids.to(dev), am.to(dev), pep.to(dev)).float().cpu().numpy()
        n_distinct = len(set(np.round(sc, 6).tolist()))
        # Both blocks are reported: the whole-molecule pool barely moves when the peptide is
        # swapped (1.3-1.9 % of its length, finding 9.14) while the peptide pool moves 7-16 %,
        # so one spread figure over the concatenation would hide which block is alive.
        spread = lambda v: float((v - v.mean(0, keepdim=True)).abs().max())
        pooled_spread = spread(torch.cat([pooled_all, pooled_pep], dim=-1))
        lens = [int(x.sum()) for x in am]
        print(f"[probe] row_id={row['row_id']} lens={lens} pooled_spread={pooled_spread:.3e} "
              f"(molecule {spread(pooled_all):.3e}, peptide {spread(pooled_pep):.3e}) "
              f"logits={np.array2string(sc, precision=6)} distinct={n_distinct}/{len(sc)}", flush=True)
        try:
            rank_of(sc)
        except ScoreError as e:
            print(f"[probe] FAIL: {e}", flush=True); ok = False
        if n_distinct < 2:
            print("[probe] FAIL: every candidate scored identically — the scorer ignores its input", flush=True); ok = False
    print(f"[probe] {'PASS' if ok else 'FAIL'}: scores {'vary' if ok else 'do not vary'} with the peptide", flush=True)
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------- paired comparison
def cmd_compare(a):
    """Thin consumer of scripts/training/stage42_box/paired_compare.py.

    What used to live here measured the wrong thing (audit finding 4, 2026-11-04): it
    intersected row IDs silently, differenced Recall@10 indicators, and bootstrapped over ROWS.
    The gate licenses `test_novel_pep / peptide / norm_rank`; a row bootstrap on a different
    metric is both the wrong estimand and an interval too narrow by construction, because
    molecules of one peptide are not independent observations.

    The statistics now live in a stdlib-only module beside readability.py, so the whole
    contract is testable without torch or peft (tests/test_paired_compare.py). This wrapper
    exists only so `judge compare` keeps working; it adds nothing of its own.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                           "scripts" / "training" / "stage42_box"))
    import paired_compare as PC

    rc, _ = PC.compare(a.a, a.b, a.units, primary=(a.pc, a.pu, a.pm), out=a.out,
                       allow_partial_pairing=a.allow_partial_pairing)
    sys.exit(rc)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval", "probe"):
        p = sp.add_parser(name)
        p.add_argument("--task", required=True, choices=["as_paired_i", "as_trb_i"])
        p.add_argument("--adapter", required=True, help="'none' = fresh LoRA (fair base), else adapter dir")
        p.add_argument("--tcr_variant", required=True)
        p.add_argument("--mhc_variant", required=True)
        p.add_argument("--max_len", type=int, default=D["max_len"])
        p.add_argument("--smoke", action="store_true")
        # No silent CPU fallback: a GPU visit that fell back produced a result nobody chose to
        # buy (audit finding 5, 2026-11-06). Opting in stays available for off-box checks.
        p.add_argument("--allow_cpu", action="store_true",
                       help="run on the CPU when CUDA is absent, instead of refusing")
    t = sp.choices["train"]
    t.add_argument("--out_dir", required=True)
    t.add_argument("--max_train_pos", type=int, default=D["max_train_pos"])
    t.add_argument("--max_val_pos", type=int, default=D["max_val_pos"])
    t.add_argument("--epochs", type=int, default=D["epochs"])
    t.add_argument("--patience", type=int, default=D["patience"])
    t.add_argument("--batch_size", type=int, default=D["batch_size"])
    t.add_argument("--lr", type=float, default=D["lr"])
    t.add_argument("--seed", type=int, default=D["seed"])
    t.add_argument("--num_workers", type=int, default=D["num_workers"])
    sp.choices["probe"].add_argument(
        "--ckpt", default=None,
        help="probe THIS checkpoint rather than a fresh head. Required to vouch for a "
             "checkpoint an evaluation will read: a fresh-model probe says nothing about it.")
    e = sp.choices["eval"]
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--conditions", nargs="+", default=["test_level4", "test_novel_pep", "test_iid"])
    e.add_argument("--max_queries", type=int, default=5000)
    # The power floor is distinct true peptides, not queries: as_paired_i test_iid has only
    # 53, so its CI is 0.33 wide however many molecules are scored (2026-10-15). The old
    # --min_queries knob was unused after that change and is gone, so nothing can pass it
    # and believe a query count still gates anything.
    e.add_argument("--min_peptides", type=int, default=100)
    # length_matched by default: a uniform draw leaks the true peptide's length against a
    # 76 %-9-mer pool (see _draw_distractors). `uniform` reproduces the 2026-10-08 cell.
    e.add_argument("--distractor_draw", choices=["length_matched", "uniform"],
                   default="length_matched")
    e.add_argument("--smoke_queries", type=int, default=20)
    c = sp.add_parser("compare")
    c.add_argument("--a", required=True, help="arm A eval.json (the base)")
    c.add_argument("--b", required=True, help="arm B eval.json (the CPT arm)")
    c.add_argument("--units", required=True,
                   help="csv: condition,row_id,peptide,pep_cluster. Required: the resampling "
                        "unit is the peptide, so the comparison cannot be computed without it.")
    c.add_argument("--out", help="write the result as json")
    c.add_argument("--primary-condition", dest="pc", default="test_novel_pep")
    c.add_argument("--primary-unit", dest="pu", default="peptide",
                   choices=["peptide", "pep_cluster"])
    c.add_argument("--primary-metric", dest="pm", default="norm_rank",
                   choices=["norm_rank", "recall_at_10", "mrr"])
    c.add_argument("--allow-partial-pairing", action="store_true",
                   help="exclude rows missing from one arm ON PURPOSE; they are counted in the "
                        "output. Without this an unequal query set refuses.")
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "probe": cmd_probe, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
