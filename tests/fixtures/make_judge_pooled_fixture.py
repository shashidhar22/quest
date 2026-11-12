#!/usr/bin/env python3
"""Regenerate `judge_pooled_dg1.npz`, the fixture behind `tests/test_judge_head_learns.py`.

The fixture holds POOLED ENCODER OUTPUTS for real `as_paired_i` molecules and their
peptide-swap negatives, so the test exercises the geometry the judge's head actually faces
rather than a synthetic stand-in. Two synthetic attempts failed to reproduce the DG1 collapse
(one made the label trivially separable, AUROC 0.999; the other got the absolute scales wrong),
which is why this is measured and committed instead of generated at test time.

What is in it, for each of `train` and `val`:
    <split>_pep     (n, 320) float16  the mean over the row's PEPTIDE tokens, projected
    <split>_label   (n,)     int8     1 for the true pairing, 0 for a peptide-swap negative
    <split>_mol     (n,)     int32    molecule index, so a test can hold out BY molecule

How it is built (this script):
  1. `judge.load_split` for the split, sampled to --n_mol molecules (one row per `row_id`).
  2. `judge.with_negatives`, the judge's own 5x peptide-swap sampler (seed 43 train / 44 val):
     the TCR and the MHC are kept and only the peptide is replaced, so ONLY the peptide
     carries the label.
  3. Each row serialized by `judge.serialize` — the judge's own path through the CPT
     trainer's `project_canonical_row_to_run_input`, `<eos>`-joined, eos_strategy "raw".
  4. ESM-C 300M, frozen, with the FRESH LoRA the fair base arm uses (`adapter="none"`), run
     through `encoders.encoder_forward`; hidden states cast to fp32 as `ESMScorer.forward` does.
  5. Tokens labelled by `cpt_training_lib.annotate_token_regions`; the mean taken over the
     tokens labelled `peptide`.
  6. Projected 960 -> 320 by a SEEDED orthonormal matrix (seed 11, the first 320 columns of
     the QR of a Gaussian), purely to keep the committed file near 1.5 MB. The projection
     preserves the conditioning the test is about; the numbers below were checked on both the
     full 960-dim vectors and the projection.

Why the peptide pooling and not the all-token pooling: both reproduce the collapse, and the
peptide pooling is the one on which the repaired head clears the prior inside the judge's own
2-epoch budget, so the fixture carries the cheaper demonstration. The all-token pooling's
numbers are in quest-cpt/findings.md 9.14.

Measured on 300 train / 150 val molecules, 2026-11-11 (laptop, MPS, ESM-C 300M):
    the 1:5 prior                       val loss 0.4506, AUROC 0.500
    unnormalized head, 2 and 12 epochs  AUROC 0.473-0.477, loss 0.4536-0.4540 over 4 seeds
    BatchNorm head, 12 epochs           AUROC 0.745-0.781, loss 0.3668-0.3869 over 4 seeds

Cost: about 15 minutes on an M-series laptop for 450 molecules (2,700 serialized rows).

    python tests/fixtures/make_judge_pooled_fixture.py --out tests/fixtures/judge_pooled_dg1.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROJECTION_SEED = 11
PROJECTION_DIM = 320


def pooled_peptide_vectors(split: str, n_mol: int, neg_seed: int, sample_seed: int,
                           device: str, batch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from track_a.esm_sft import judge as J
    from track_a.esm_sft.encoders import encoder_forward

    df = J.load_split("as_paired_i", split)
    df = df.sample(n=min(n_mol, len(df)), random_state=sample_seed).reset_index(drop=True)
    rows = J.with_negatives(df, neg_seed)
    mol_of = {rid: i for i, rid in enumerate(df["row_id"])}

    model, tok = J.build_model("none", torch.device("cpu"))
    model = model.to(torch.device(device)).eval()
    loaded = model.loaded
    records = [J._clean(r) for r in rows.to_dict("records")]

    vecs, labels, mols = [], [], []
    t0 = time.time()
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        seqs = [J.serialized_or_refuse(r, "cdr3", "full", tok, 1024, "fixture row")
                for r in chunk]
        ids, am, peptide_mask = J.pad(seqs, int(tok.pad_token_id))
        with torch.no_grad():
            h = encoder_forward(loaded, ids.to(device), am.to(device)).float().cpu().numpy()
        mask = peptide_mask.numpy().astype(bool) & am.numpy().astype(bool)
        for j, r in enumerate(chunk):
            is_pep = mask[j]
            vecs.append(h[j][is_pep].mean(0))
            labels.append(int(r["label"]))
            mols.append(mol_of[r["row_id"]])
        print(f"  {split}: {start + len(chunk)}/{len(records)} rows, {time.time() - t0:.0f}s",
              flush=True)
    return np.asarray(vecs), np.asarray(labels, dtype=np.int8), np.asarray(mols, dtype=np.int32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).with_name("judge_pooled_dg1.npz")))
    ap.add_argument("--n_mol_train", type=int, default=300)
    ap.add_argument("--n_mol_val", type=int, default=150)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--splits_dir", default=None,
                    help="QUEST_SPLITS_DIR; the committed fixture used the DG1 audit copy")
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        ap.error(f"{a.out} exists; a fixture is replaced deliberately, not by accident")
    if a.splits_dir:
        os.environ["QUEST_SPLITS_DIR"] = a.splits_dir
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts" / "training"))

    rng = np.random.default_rng(PROJECTION_SEED)
    out = {}
    for split, n_mol, neg_seed, sample_seed in (("train", a.n_mol_train, 43, 1),
                                                ("val", a.n_mol_val, 44, 2)):
        X, y, mol = pooled_peptide_vectors(split, n_mol, neg_seed, sample_seed, a.device, a.batch)
        if "P" not in out:
            # built once, from the encoder width, so both splits share one projection
            out["P"] = np.linalg.qr(rng.normal(size=(X.shape[1], X.shape[1])))[0][:, :PROJECTION_DIM]
        out[f"{split}_pep"] = (X.astype(np.float64) @ out["P"]).astype(np.float16)
        out[f"{split}_label"] = y
        out[f"{split}_mol"] = mol
    out.pop("P")
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
