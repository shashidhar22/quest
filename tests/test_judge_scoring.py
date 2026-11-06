"""What a score vector is allowed to become.

Audit finding 2, 2026-11-06. `judge.cmd_eval` computed a rank as

    1 + (sc[1:] >= sc[0]).sum()

with no finiteness check. Every comparison against NaN is False, so a query whose logits came
back all-NaN scored rank 1 -- a PERFECT retrieval -- and the downstream rank validation in the
gate and the comparator cannot see it, because 1 is a legal rank. A silent numerical failure
would therefore have been read as the best possible result, in the one number the visit exists
to produce.

The second half is the checkpoint. The evaluator merged the saved trainable tensors into a FRESH
state dict, so any trained key absent from the file kept its freshly initialised value and the
load still succeeded. A checkpoint with no head, or with the head only and no LoRA, evaluated as
if it were trained.

Both are pure logic over a score vector and two key sets, so they live in a module with no torch
in its import chain and are tested here without the ML stack, including in a subprocess where
torch is blocked outright.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from track_a.esm_sft.scoring import (  # noqa: E402
    ScoreError,
    missing_trainable_keys,
    rank_of,
    score_diagnostics,
)


# ----------------------------------------------------------------- the defect itself
def test_an_all_nan_query_is_refused_rather_than_ranked_first():
    """The exact reproduction from the audit: 100 NaNs used to produce rank 1."""
    with pytest.raises(ScoreError, match="non-finite"):
        rank_of(np.full(100, np.nan))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_one_non_finite_candidate_anywhere_refuses(bad):
    sc = np.linspace(1.0, 0.0, 100)
    sc[37] = bad
    with pytest.raises(ScoreError, match="non-finite"):
        rank_of(sc)
    sc = np.linspace(1.0, 0.0, 100)
    sc[0] = bad          # the true peptide itself
    with pytest.raises(ScoreError, match="non-finite"):
        rank_of(sc)


def test_the_historical_rank_semantics_are_unchanged():
    """The repair must not move a number: ties count AGAINST the true peptide.

    Index 0 is the true peptide. This is the 2026-10-08 convention and the gate, the
    readability predicate and the comparator all read ranks produced under it.
    """
    assert rank_of(np.array([0.9, 0.5, 0.1])) == 1
    assert rank_of(np.array([0.5, 0.9, 0.1])) == 2
    assert rank_of(np.array([0.1, 0.9, 0.5])) == 3
    assert rank_of(np.array([0.5, 0.5, 0.5])) == 3, "a full tie is the worst rank, not the best"
    assert rank_of(np.array([0.5, 0.5, 0.1])) == 2


def test_a_degenerate_candidate_set_is_refused():
    """A query with nothing to rank against is a fault, not a rank-1 result."""
    with pytest.raises(ScoreError, match="at least two"):
        rank_of(np.array([0.5]))
    with pytest.raises(ScoreError, match="at least two"):
        rank_of(np.array([]))


# ----------------------------------------------------------------- the diagnostics
def test_diagnostics_separate_a_dead_scorer_from_a_working_one():
    """Wave 1 burned 18 GPU-hours on a scorer whose candidates all tied. A rank cannot show
    that on its own -- an all-tie query and a genuinely last-placed one both score rank N --
    so the spread and the tie count are recorded beside the rank."""
    dead = score_diagnostics(np.full(100, 0.25))
    assert dead["spread"] == 0.0
    assert dead["n_distinct"] == 1
    assert dead["all_tied"] is True

    alive = score_diagnostics(np.linspace(0.0, 1.0, 100))
    assert alive["spread"] == pytest.approx(1.0)
    assert alive["n_distinct"] == 100
    assert alive["all_tied"] is False


def test_diagnostics_do_not_raise_on_non_finite_scores():
    """They are evidence about a failure, so they must survive one and report it."""
    d = score_diagnostics(np.array([np.nan, 1.0, 2.0]))
    assert d["n_non_finite"] == 1
    assert d["all_tied"] is False


# ----------------------------------------------------------------- the checkpoint
MODEL_KEYS = (
    "loaded.base.layer0.weight",            # frozen, not expected in a checkpoint
    "pool.q.weight", "pool.q.bias",
    "head.0.weight", "head.0.bias",
    "loaded.base.layer0.lora_A.weight", "loaded.base.layer0.lora_B.weight",
)
PATTERNS = ("pool.", "head.", "lora_")


def test_a_checkpoint_missing_its_trained_head_is_refused():
    """The silent-merge defect. `sd.update(saved)` over a fresh state dict means an absent
    trained key keeps its random init and `load_state_dict` still succeeds."""
    saved = [k for k in MODEL_KEYS if k.startswith("pool.") or "lora_" in k]
    missing, unexpected = missing_trainable_keys(saved, MODEL_KEYS, PATTERNS)
    assert sorted(missing) == ["head.0.bias", "head.0.weight"]
    assert unexpected == []


def test_a_checkpoint_with_no_lora_is_refused():
    saved = [k for k in MODEL_KEYS if k.startswith(("pool.", "head."))]
    missing, _ = missing_trainable_keys(saved, MODEL_KEYS, PATTERNS)
    assert all("lora_" in k for k in missing) and missing


def test_a_complete_checkpoint_passes_and_frozen_weights_are_not_wanted():
    saved = [k for k in MODEL_KEYS if any(p in k for p in PATTERNS)]
    assert missing_trainable_keys(saved, MODEL_KEYS, PATTERNS) == ([], [])
    # the frozen base is deliberately absent from a checkpoint and must not be demanded
    assert "loaded.base.layer0.weight" not in saved


def test_a_key_the_model_does_not_have_is_reported():
    saved = [k for k in MODEL_KEYS if any(p in k for p in PATTERNS)] + ["head.9.weight"]
    missing, unexpected = missing_trainable_keys(saved, MODEL_KEYS, PATTERNS)
    assert missing == []
    assert unexpected == ["head.9.weight"]


def test_an_empty_checkpoint_is_refused_loudly():
    missing, _ = missing_trainable_keys([], MODEL_KEYS, PATTERNS)
    assert len(missing) == 6, "every trainable key is missing from an empty checkpoint"


# ----------------------------------------------------------------- the import boundary
def test_the_scoring_rules_import_without_torch():
    """Proved, not assumed: last round's lesson was that a light module under a heavy package
    is still heavy. judge.py imports cpt_training_lib, hence peft and torch, so these rules
    could not be exercised on a laptop while they lived there."""
    code = textwrap.dedent(
        r"""
        import importlib.abc
        import sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                for heavy in ("torch", "peft", "esm", "pandas", "transformers"):
                    if fullname == heavy or fullname.startswith(heavy + "."):
                        raise ImportError(heavy + " is blocked for this test")
                return None

        sys.meta_path.insert(0, Block())
        import numpy as np
        from track_a.esm_sft.scoring import ScoreError, rank_of

        try:
            rank_of(np.full(100, np.nan))
        except ScoreError:
            print("OK")
        else:
            raise AssertionError("100 NaNs produced a rank")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (
        "importing the scoring rules reached the ML stack:\n" + r.stdout + r.stderr
    )


# ----------------------------------------------------------------- the wiring
# These are source-level checks, which is weaker than exercising the function. They are here
# because `judge.cmd_eval` cannot be called on a laptop at all: it imports the CPT trainer,
# builds an ESM-C model and needs a GPU. The auditor found the defect the same way, by AST
# extraction, so the same reading is the available regression. A test that cannot fail proves
# nothing, so each one names what it would catch.
JUDGE_SRC = (REPO / "track_a" / "esm_sft" / "judge.py").read_text()


def test_the_evaluator_no_longer_computes_a_rank_by_hand():
    """Catches a reintroduced `1 + (sc[1:] >= sc[0]).sum()` anywhere in the judge."""
    assert "sc[1:] >= sc[0]" not in JUDGE_SRC, (
        "the raw rank comparison is back; a non-finite score would rank 1 again")
    assert "ranks.append(rank_of(sc))" in JUDGE_SRC


def test_the_evaluator_no_longer_merges_a_checkpoint_over_fresh_weights():
    """Catches a reintroduced `sd.update(ck["trainable_state_dict"])`, which let a checkpoint
    missing its trained head evaluate as if trained."""
    assert 'sd.update(ck["trainable_state_dict"])' not in JUDGE_SRC
    assert "load_checkpoint(model, a.ckpt, dev, want=vars(a))" in JUDGE_SRC, (
        "want=vars(a) is what makes the checkpoint's recorded settings checked against the "
        "evaluation asking for it")


def test_the_probe_can_vouch_for_a_real_checkpoint():
    """The audit's point: a probe of a freshly built model says nothing about the checkpoint the
    paid evaluation reads."""
    assert 'sp.choices["probe"].add_argument(' in JUDGE_SRC
    assert JUDGE_SRC.count("load_checkpoint(model, a.ckpt, dev, want=vars(a))") == 2, (
        "both the probe and the evaluation must validate the checkpoint they load")
    assert "vouches for no checkpoint" in JUDGE_SRC, (
        "a fresh-head probe must say so in its own output")


# ----------------------------------------------------------------- checkpoint compatibility
def test_a_checkpoint_trained_for_another_task_is_refused():
    from track_a.esm_sft.scoring import incompatible_checkpoint_args as f

    trained = {"task": "as_trb_i", "tcr_variant": "cdr3", "mhc_variant": "full",
               "adapter": "none", "max_len": 1024}
    asked = dict(trained, task="as_paired_i")
    bad = f(trained, asked)
    assert len(bad) == 1 and "task" in bad[0], bad
    assert f(trained, dict(trained)) == []


def test_a_checkpoint_evaluated_through_a_different_adapter_is_refused():
    """The head was trained on one encoder init. Reading it through another mixes two arms."""
    from track_a.esm_sft.scoring import incompatible_checkpoint_args as f

    trained = {"task": "as_paired_i", "tcr_variant": "cdr3", "mhc_variant": "full",
               "adapter": "none", "max_len": 1024}
    bad = f(trained, dict(trained, adapter="checkpoints/cpt_stage41_pointD_run05/final"))
    assert len(bad) == 1 and "adapter" in bad[0], bad


def test_a_checkpoint_that_recorded_nothing_is_not_treated_as_agreeing():
    from track_a.esm_sft.scoring import incompatible_checkpoint_args as f

    bad = f({}, {"task": "as_paired_i", "tcr_variant": "cdr3", "mhc_variant": "full",
                 "adapter": "none", "max_len": 1024})
    assert len(bad) == 5
    assert all("records nothing" in b for b in bad)
    bad = f(None, {"task": "as_paired_i"})
    assert len(bad) == 5, "a checkpoint with no args block at all is not compatible by default"


# ----------------------------------------------------------------- round two, finding 4
def test_the_sft_recipe_comes_from_the_checkpoints_own_args():
    from track_a.esm_sft.scoring import UNRECORDED, sft_recipe

    full = {"lr": 1e-4, "batch_size": 16, "epochs": 2, "patience": 1,
            "max_train_pos": 50_000, "max_val_pos": 5_000, "num_workers": 4, "seed": 42}
    got = sft_recipe(full)
    assert got != UNRECORDED
    assert "lr=0.0001" in got and "batch_size=16" in got and "max_train_pos=50000" in got
    assert sft_recipe(dict(full)) == got, "the recipe string must be deterministic"
    assert sft_recipe(dict(full, lr=1e-3)) != got


def test_a_partially_recorded_recipe_is_unrecorded_not_partial():
    """A half-known recipe cannot be shown equal to another, and provenance refuses the
    sentinel -- so reporting the known half would be worse than reporting nothing."""
    from track_a.esm_sft.scoring import UNRECORDED, sft_recipe

    full = {"lr": 1e-4, "batch_size": 16, "epochs": 2, "patience": 1,
            "max_train_pos": 50_000, "max_val_pos": 5_000, "num_workers": 4}
    for drop in full:
        partial = {k: v for k, v in full.items() if k != drop}
        assert sft_recipe(partial) == UNRECORDED, drop
    assert sft_recipe(None) == UNRECORDED
    assert sft_recipe({}) == UNRECORDED


def test_the_evaluator_keeps_the_checkpoint_it_loaded():
    """The defect: `load_checkpoint(...)`'s return value was discarded, and the SFT seed was
    read from the eval's own namespace -- where --seed does not exist, so it was always None."""
    assert "ck = load_checkpoint(model, a.ckpt, dev, want=vars(a))" in JUDGE_SRC
    # the defect was the assignment, not the mention: the comment explaining it stays
    code = "\n".join(l for l in JUDGE_SRC.splitlines() if not l.lstrip().startswith("#"))
    assert 'getattr(a, "seed"' not in code
    assert '"sft_seed": ck_args.get("seed")' in JUDGE_SRC
    assert '"sft_recipe": sft_recipe(ck_args)' in JUDGE_SRC


def test_the_eval_parser_still_has_no_seed_of_its_own():
    """Which is why reading one off it was always going to be null. If a --seed is ever added to
    eval, this test fails and whoever adds it has to decide what it means for lineage."""
    block = JUDGE_SRC[JUDGE_SRC.index('e = sp.choices["eval"]'):
                      JUDGE_SRC.index('c = sp.add_parser("compare")')]
    assert "--seed" not in block


def test_training_hashes_the_data_it_trained_on():
    assert 'A.split_digests(a.task, ("train", "val"))' in JUDGE_SRC
    assert '"data_digests"' in JUDGE_SRC


def test_the_cotenancy_record_is_read_not_declared():
    from track_a.esm_sft.scoring import cotenants
    import os

    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        assert cotenants() == "cuda_visible_devices=3"
        del os.environ["CUDA_VISIBLE_DEVICES"]
        assert cotenants() == "cuda_visible_devices=unset"
    finally:
        if old is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
