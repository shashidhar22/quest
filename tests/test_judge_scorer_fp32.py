"""The Stage 4.2 judge's scorer must resolve small differences between retrieval candidates.

On 2026-10-07 wave-1 cell 1 scored every one of 15,000 queries with all 100 candidates tied
(Recall@10 = 0.0, MRR = 1/100, AUC = 0.5 exactly): the pool + head ran under bf16 autocast, and
candidates that differ only in a 9-residue peptide out of ~400 tokens produced logits within one
bf16 ulp. These tests use a stub encoder (no ESM-C needed) and CPU bf16 autocast to show the
collapse mechanism and that the fp32 head resolves it.
"""
import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from track_a.esm_sft.encoders import LoadedEncoder  # noqa: E402
from track_a.esm_sft.model import ESMScorer  # noqa: E402

H = 64


class _StubEncoder(nn.Module):
    """Returns per-token embeddings that depend on the token ids only slightly: a fixed vector
    plus a tiny token-dependent perturbation, mimicking a frozen encoder whose pooled output
    barely moves when 9 of 400 tokens change."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.base = nn.Parameter(torch.randn(H), requires_grad=False)
        self.emb = nn.Embedding(64, H)

    def forward(self, input_ids=None, attention_mask=None, **_):
        # encoder_forward's plain-AutoModel path reads `.last_hidden_state`
        return types.SimpleNamespace(last_hidden_state=self.base + self.scale * self.emb(input_ids))


def _scorer(scale):
    torch.manual_seed(0)
    enc = _StubEncoder(scale)
    loaded = LoadedEncoder(encoder=enc, tokenizer=None, hidden_dim=H, is_esmc=False)
    return ESMScorer(loaded, head_dim=32, dropout=0.0, freeze_base=True, peptide_pool=True).eval()


def _candidates(n=100, L=40, differ_at=5):
    """n sequences identical except for `differ_at` positions (the 'peptide').

    Returns ``(ids, attention mask, peptide mask)``. The peptide mask marks exactly the
    positions that vary, which is what the real serializer's region labels mark and what
    `ESMScorer` pools its peptide block over.
    """
    torch.manual_seed(1)
    base = torch.randint(4, 24, (1, L))
    ids = base.repeat(n, 1)
    ids[:, 10:10 + differ_at] = torch.randint(4, 24, (n, differ_at))
    ids = torch.unique(ids, dim=0)          # identical inputs may legitimately tie; drop them
    pep = torch.zeros(len(ids), L, dtype=torch.long)
    pep[:, 10:10 + differ_at] = 1
    return ids, torch.ones(len(ids), L, dtype=torch.long), pep


def test_fp32_head_resolves_candidates_that_differ_slightly():
    """Failure: the new forward ties most candidates whose pooled vectors differ by ~1e-5
    relative. (Not 'all distinct': at this perturbation two different inputs can land within one
    fp32 ulp by chance; the bf16 path below resolves at most a handful of 100.)"""
    m = _scorer(scale=1e-3)
    ids, am, pep = _candidates()
    with torch.no_grad():
        sc = m(ids, am, pep)
    assert sc.dtype == torch.float32
    assert torch.isfinite(sc).all()
    n_distinct = len(set(sc.tolist()))
    assert n_distinct >= 0.9 * len(sc), f"only {n_distinct}/{len(sc)} distinct logits under the fp32 head"


def test_bf16_pool_and_head_would_tie_them():
    """Documents the mechanism and keeps the fix honest: the SAME pooled vectors pushed through
    the pool + head under bf16 autocast collapse to a handful of values. Failure here would mean
    the stub no longer reproduces the field failure, i.e. the test above proves less than claimed."""
    m = _scorer(scale=1e-3)
    ids, am, pep = _candidates()
    with torch.no_grad():
        h = m.encoder(ids, am).last_hidden_state.float()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):      # the old arrangement
            pooled = torch.cat([m.pool(h, am.bool()), m.pool_peptide(h, pep.bool())], dim=-1)
            old = m.head(pooled).squeeze(-1).float()
        new = m(ids, am, pep)
    n_old, n_new = len(set(old.tolist())), len(set(new.tolist()))
    assert n_old < n_new, (n_old, n_new)
    assert n_old <= 8, f"bf16 path still resolved {n_old} of {len(ids)} candidates — the stub is too coarse"


def test_forward_runs_pool_and_head_outside_autocast_on_cpu_and_returns_fp32():
    """Failure: logits come back in a reduced dtype, or an autocast context leaks into the head."""
    m = _scorer(scale=1e-2)
    ids, am, pep = _candidates(n=4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with torch.no_grad():
            sc = m(ids, am, pep)
    assert sc.dtype == torch.float32
    assert len(set(sc.tolist())) == 4


def test_legacy_track_a_scorer_still_accepts_two_arguments():
    """The DG1 repair must not change Track A's checkpoint shape or call contract."""
    enc = _StubEncoder(0.01)
    loaded = LoadedEncoder(encoder=enc, tokenizer=None, hidden_dim=H, is_esmc=False)
    model = ESMScorer(loaded, head_dim=32, dropout=0.0).eval()
    ids, am, _ = _candidates(n=4)
    with torch.no_grad():
        scores = model(ids, am)
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert model.state_dict()['head.0.weight'].shape == (32, H)
