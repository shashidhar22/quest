"""The judge's head must learn the peptide signal that is measurably in its input.

Visit DG1 (2026-11-11) found both V04a SFT judges returning one logit for every candidate:
every score within 1e-04 of -1.669, against the 1:5 training prior's own -1.60944, and a
training loss of 0.4510 where predicting the prior costs 0.450561. The judge had learned the
base rate and was ranking on leftovers, so the comparison built on it compared two constants.

WHY THIS TEST USES A MEASURED FIXTURE rather than synthetic vectors. Two synthetic attempts
failed, in opposite directions: the first made the label trivially separable (an unnormalized
control reached AUROC 0.999), the second got the absolute scales wrong. The failure depends on
the real geometry of ESM-C's pooled output, so the fixture holds real pooled vectors --
`tests/fixtures/judge_pooled_dg1.npz`, built by the generator beside it, which records exactly
how. Both halves of the finding reproduce on it, stably across seeds.

WHAT THE FIXTURE IS: for 300 training and 150 validation `as_paired_i` molecules, each with its
five peptide-swap negatives, the mean over the row's PEPTIDE tokens of the frozen encoder's
output, projected 960 -> 320 by a seeded orthonormal matrix to keep the committed file small.
Positives and negatives inside one molecule share the TCR and the MHC, so only the peptide
carries the label, exactly as in training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FIXTURE = Path(__file__).with_name("fixtures") / "judge_pooled_dg1.npz"
# -[p log p + (1-p) log(1-p)] at p = 1/6, the 1:5 negative ratio. The number to beat: a judge
# that does not is predicting the base rate, which is finding 9.13's signature.
PRIOR_LOSS = 0.450561
EPOCHS = 2


def _fixture():
    if not FIXTURE.exists():
        raise AssertionError(
            f"{FIXTURE} is missing. It is committed, not generated at test time; rebuild it with "
            f"tests/fixtures/make_judge_pooled_fixture.py (about 15 minutes on a laptop). This "
            f"test is not skipped when it is absent, because a silent skip here would let the "
            f"DG1 collapse back in unnoticed.")
    return np.load(FIXTURE)


def _train_and_score(head, d, epochs=EPOCHS, seed=0):
    """Train `head` on the fixture's training rows with the judge's own recipe; read the
    validation rows. AdamW, lr 1e-4, weight decay 1e-5, BCE, batch 16 -- `SFT_TRAIN_DEFAULTS`.

    Returns (AUROC, mean BCE, within-molecule pairwise accuracy) on held-out molecules.
    """
    from sklearn.metrics import log_loss, roc_auc_score

    from track_a.esm_sft.scoring import SFT_TRAIN_DEFAULTS as D

    torch.manual_seed(seed)
    Xtr = torch.from_numpy(d["train_pep"].astype(np.float32))
    ytr = torch.from_numpy(d["train_label"].astype(np.float32))
    Xva = torch.from_numpy(d["val_pep"].astype(np.float32))
    yva = d["val_label"].astype(int)
    opt = torch.optim.AdamW(head.parameters(), lr=D["lr"], weight_decay=1e-5)
    bce = torch.nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        head.train()
        for idx in torch.randperm(len(ytr), generator=g).split(D["batch_size"]):
            if len(idx) < 2:        # BatchNorm1d needs more than one row in train mode
                continue
            loss = bce(head(Xtr[idx]).squeeze(-1), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        logit = head(Xva).squeeze(-1).double().numpy()
    p = 1.0 / (1.0 + np.exp(-logit))
    within = []
    for mol in np.unique(d["val_mol"]):
        idx = d["val_mol"] == mol
        positive, negative = logit[idx & (yva == 1)], logit[idx & (yva == 0)]
        within.append(np.mean((positive > negative) + 0.5 * (positive == negative)))
    return roc_auc_score(yva, p), log_loss(yva, p, labels=[0, 1]), float(np.mean(within))


def test_the_fixture_still_holds_the_geometry_the_finding_describes():
    """The fixture is evidence, so its shape is asserted rather than assumed.

    If this fails, the fixture was rebuilt from different data and the thresholds below were
    calibrated against something else: re-measure before trusting either test.
    """
    d = _fixture()
    for split, n_mol in (("train", 300), ("val", 150)):
        label, mol = d[f"{split}_label"], d[f"{split}_mol"]
        assert len(np.unique(mol)) == n_mol, f"{split}: {len(np.unique(mol))} molecules"
        # one positive to five negatives, and the positive is not identifiable from the molecule
        assert abs(label.mean() - 1 / 6) < 0.01, f"{split}: positive rate {label.mean():.4f}"
        counts = {int((mol == m).sum()) for m in np.unique(mol)}
        assert counts == {6}, f"{split}: rows per molecule {sorted(counts)}"
    X = d["train_pep"].astype(np.float64)
    mol = d["train_mol"]
    centroids = np.stack([X[mol == m].mean(0) for m in np.unique(mol)])
    within = np.concatenate([X[mol == m] - X[mol == m].mean(0) for m in np.unique(mol)])
    ratio = within.var(0).sum() / centroids.var(0).sum()
    assert 0.3 < ratio < 1.5, (
        f"the peptide swap moves the pooled vector by {ratio:.3f} of the between-molecule "
        f"spread; the committed fixture measured 0.75")


def test_the_judges_head_learns_the_peptide_signal():
    """`build_head` — per-feature normalization, then the MLP — must beat the base rate.

    Rechecked at the two-epoch recipe budget: AUROC 0.687-0.725, loss 0.409-0.427
    over four seeds. This is a frozen-feature test, not an end-to-end judge result.
    """
    from track_a.esm_sft.model import build_head

    d = _fixture()
    torch.manual_seed(0)
    auroc, loss, within = _train_and_score(build_head(d["train_pep"].shape[1], head_dim=256, dropout=0.1), d)
    assert loss < PRIOR_LOSS, (
        f"held-out loss {loss:.4f} is not below the 1:5 prior's {PRIOR_LOSS}: the head learned "
        f"the base rate and nothing else, which is finding 9.13 returning")
    assert within > 0.60, f"within-molecule pairwise accuracy {within:.4f}: query offsets are insufficient"
    assert auroc > 0.65, (
        f"held-out AUROC {auroc:.4f}: the head cannot rank, so it cannot rank 100 retrieval "
        f"candidates either")


def test_an_unnormalized_head_is_the_failure_this_guards_against():
    """The control: same fixture, same recipe, no normalization of the pooled vector.

    This is the V04a architecture, and it must FAIL, so the guard above cannot pass for an
    unrelated reason. Measured over four seeds: AUROC 0.473-0.477, loss 0.4536-0.4540 — never
    below the prior. Note what this does NOT claim: the signal is reachable without
    normalization given a large enough weight norm (logistic regression on these same raw
    features reaches AUROC 0.757 once its L2 penalty is relaxed to C=100). What normalization
    buys is reaching it inside this recipe's step budget.
    """
    d = _fixture()
    H = d["train_pep"].shape[1]
    torch.manual_seed(0)
    bare = torch.nn.Sequential(
        torch.nn.Linear(H, 256), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, 1))
    auroc, loss, within = _train_and_score(bare, d)
    assert loss >= PRIOR_LOSS - 1e-3 and auroc < 0.60, (
        f"the unnormalized head reached AUROC {auroc:.4f} and loss {loss:.4f}. It is supposed to "
        f"sit at the base rate here; if it no longer does, the fixture or the recipe changed and "
        f"the thresholds above are calibrated against something else")
