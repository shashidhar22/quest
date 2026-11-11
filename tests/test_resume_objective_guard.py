"""Resume must refuse an objective change, whichever objective is finally adopted.

From the 2026-11-04 audit and its follow-up. `check_resume_config` compared 26 fields
and `loss_weighting` was not among them, while `_apply_resume_state` restored optimizer
and scheduler state: a row-share checkpoint resumed into a token-share run silently,
carrying the optimizer with it.

Refusing an A-to-B change is correct under either candidate objective, so the guard does
not wait on the scientific decision. The semantics string is versioned because the
decision on the table changes what `token` *means* without changing the word: today it
weights a micro-batch by its input-length share while the loss is a mean over realized
masked targets, and an exact global masked-target mean would be a different objective
under the same name.

These tests import a module with no torch or peft dependency, so they run anywhere.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from quest.training.resume_compat import (  # noqa: E402
    LOSS_WEIGHTING_SEMANTICS,
    check_resume_config,
    objective_fingerprint,
)

# A minimal config pair that agrees on everything the generic key check looks at.
BASE = {
    "base_model": "esmc_300m",
    "dataset_path": "/d/run05",
    "canonical_eval_path": "/d/canon",
    "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05,
    "lora_target_modules": ["layernorm_qkv.1", "out_proj"],
    "learning_rate": 2e-4, "weight_decay": 0.01,
    "adam_beta1": 0.9, "adam_beta2": 0.999,
    "warmup_ratio": 0.06, "gradient_clip": 1.0,
    "num_epochs": 1, "max_length": 1024, "mlm_probability": 0.15,
    "eos_strategy": "raw", "token_budget": 16384,
    "bucket_boundaries": [64, 128, 256, 512, 1024],
    "use_length_bucketing": True, "region_eval": True,
    "tcr_variant": "cdr3", "mhc_variant": "full",
    "early_stop_metric_path": "canonical/pocket/ppl",
    "early_stop_n_patience": 0, "early_stop_min_rel": 0.01,
    "seed": 1, "rows_per_step": 1024,
    "loss_weighting": "token",
    "loss_weighting_semantics": LOSS_WEIGHTING_SEMANTICS["token"],
}


def cfg(**over):
    d = dict(BASE)
    d.update(over)
    return d


def test_identical_objective_resumes():
    check_resume_config(cfg(), cfg(), token_mode=True)


def test_row_to_token_is_refused():
    saved = cfg(loss_weighting="row",
                loss_weighting_semantics=LOSS_WEIGHTING_SEMANTICS["row"])
    with pytest.raises(ValueError, match="objective"):
        check_resume_config(saved, cfg(), token_mode=True)


def test_token_to_row_is_refused():
    current = cfg(loss_weighting="row",
                  loss_weighting_semantics=LOSS_WEIGHTING_SEMANTICS["row"])
    with pytest.raises(ValueError, match="objective"):
        check_resume_config(cfg(), current, token_mode=True)


def test_same_word_different_semantics_is_refused():
    """The decision on the table redefines `token` without renaming it."""
    current = cfg(loss_weighting_semantics="global-masked-target-mean/v2")
    with pytest.raises(ValueError, match="objective"):
        check_resume_config(cfg(), current, token_mode=True)


def test_checkpoint_without_objective_metadata_fails_closed():
    """Absent metadata must not inherit the current objective by default."""
    saved = cfg()
    del saved["loss_weighting"]
    del saved["loss_weighting_semantics"]
    with pytest.raises(ValueError) as e:
        check_resume_config(saved, cfg(), token_mode=True)
    assert "assume_saved_objective" in str(e.value), (
        "the refusal must name the explicit migration route"
    )


def test_unversioned_checkpoint_fails_closed():
    """A checkpoint from before versioning states a word but not its meaning."""
    saved = cfg()
    del saved["loss_weighting_semantics"]
    with pytest.raises(ValueError, match="objective"):
        check_resume_config(saved, cfg(), token_mode=True)


def test_explicit_assumption_admits_an_unversioned_checkpoint():
    """The operator may state what the checkpoint is, and it is then checked."""
    saved = cfg()
    del saved["loss_weighting_semantics"]
    check_resume_config(
        saved, cfg(), token_mode=True,
        assume_saved_objective=f"token:{LOSS_WEIGHTING_SEMANTICS['token']}",
    )


def test_wrong_explicit_assumption_is_refused():
    """An assumption that contradicts the checkpoint is worse than none."""
    saved = cfg(loss_weighting="row",
                loss_weighting_semantics=LOSS_WEIGHTING_SEMANTICS["row"])
    with pytest.raises(ValueError, match="does not match the checkpoint"):
        check_resume_config(
            saved, cfg(), token_mode=True,
            assume_saved_objective=f"token:{LOSS_WEIGHTING_SEMANTICS['token']}",
        )


def test_assumption_cannot_license_a_real_objective_change():
    """Stating the checkpoint's objective correctly still does not permit switching."""
    saved = cfg(loss_weighting="row",
                loss_weighting_semantics=LOSS_WEIGHTING_SEMANTICS["row"])
    with pytest.raises(ValueError, match="objective"):
        check_resume_config(
            saved, cfg(), token_mode=True,
            assume_saved_objective=f"row:{LOSS_WEIGHTING_SEMANTICS['row']}",
        )


def test_objective_fingerprint_is_word_and_meaning():
    assert objective_fingerprint(cfg()) == f"token:{LOSS_WEIGHTING_SEMANTICS['token']}"


# --------------------------------------------------- the pre-existing contract still holds
def test_unrelated_config_drift_is_still_refused():
    with pytest.raises(ValueError, match="learning_rate"):
        check_resume_config(cfg(), cfg(learning_rate=1e-4), token_mode=True)


def test_gpu_count_and_cadence_changes_are_still_allowed_in_token_mode():
    """The token-budget sampler is world-size invariant, so these may differ."""
    check_resume_config(
        cfg(), cfg(world_size=4, batch_size=8, gradient_accumulation_steps=128,
                   eval_steps=250, output_dir="/elsewhere"),
        token_mode=True,
    )


def test_legacy_path_still_pins_batch_shape():
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        check_resume_config(
            cfg(batch_size=16, gradient_accumulation_steps=64, world_size=4),
            cfg(batch_size=16, gradient_accumulation_steps=32, world_size=4),
            token_mode=False,
        )


# --------------------------------------------------- the import contract is part of the point
def test_the_rule_imports_with_no_ml_stack_present():
    """Extracting the rule only helps if it can be read without torch installed.

    The first version of this extraction did not achieve that: the module itself is
    light, but `quest/training/__init__.py` eagerly re-exported collators, which import
    torch, so `import quest.training.resume_compat` pulled the whole ML stack in. It
    passed here only because this machine's default interpreter happens to have torch;
    on a torch-less one it raised ModuleNotFoundError at collection.

    A fresh subprocess with torch blocked is the falsifiable form: if any import in the
    chain reaches torch, this fails.
    """
    code = textwrap.dedent(
        r"""
        import importlib.abc
        import sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise ImportError("torch is blocked for this test")
                return None

        sys.meta_path.insert(0, Block())

        from quest.training.resume_compat import check_resume_config, objective_fingerprint

        assert objective_fingerprint(
            {"loss_weighting": "token", "loss_weighting_semantics": "input-length-share/v1"}
        ) == "token:input-length-share/v1"

        try:
            check_resume_config(
                {"loss_weighting": "row", "loss_weighting_semantics": "row-share/v1"},
                {"loss_weighting": "token", "loss_weighting_semantics": "input-length-share/v1"},
                token_mode=True,
            )
        except ValueError as e:
            assert "objective" in str(e), e
        else:
            raise AssertionError("row-share resumed into token-share without a refusal")
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (
        "importing the resume rule reached torch:\n" + r.stdout + r.stderr
    )


def test_the_training_package_still_exposes_its_documented_names():
    """Laziness must not break `from quest.training import EarlyStopping`.

    Those re-exports are documented in CLAUDE.md, so making the package light has to
    preserve them rather than delete them.
    """
    pytest.importorskip("torch", reason="the eager re-exports legitimately need torch")
    import importlib

    mod = importlib.import_module("quest.training")
    for name in ("EarlyStopping", "StreamingMetricsTracker", "DataCollatorForMLMDynamic",
                 "LengthBucketSampler", "BaseTCRTrainer", "get_backend"):
        assert hasattr(mod, name), f"quest.training no longer exposes {name}"
    assert name in dir(mod), "dir() must still list the re-exports"


def test_the_lazy_export_table_and_all_agree():
    """A literal __all__ drifts from the lazy table unless something checks."""
    import importlib

    mod = importlib.import_module("quest.training")
    assert sorted(mod.__all__) == sorted(mod._EXPORTS), (
        "quest.training.__all__ and _EXPORTS disagree: "
        f"{sorted(set(mod.__all__) ^ set(mod._EXPORTS))}"
    )


IDENTITY = {"run_id": "r1", "contract_sha256": "c" * 64, "contract_version": 4,
            "marker_sha256": "m" * 64, "benchmark_checked": False}


def test_a_resume_on_another_corpus_at_the_same_path_is_refused():
    """`dataset_path` matching says nothing when a corpus was republished there in between."""
    saved = cfg(corpus_identity=dict(IDENTITY))
    check_resume_config(saved, cfg(corpus_identity=dict(IDENTITY)), token_mode=True)
    # Whether the benchmark hashes were also checked is about the audit, not which corpus it is.
    check_resume_config(saved, cfg(corpus_identity={**IDENTITY, "benchmark_checked": True}), token_mode=True)
    for change in ({"run_id": "r2"}, {"marker_sha256": "n" * 64}, {"contract_sha256": "d" * 64}):
        with pytest.raises(ValueError, match="corpus_identity"):
            check_resume_config(saved, cfg(corpus_identity={**IDENTITY, **change}), token_mode=True)
    # Recorded on one side only is not agreement, in either direction.
    with pytest.raises(ValueError, match="corpus_identity"):
        check_resume_config(saved, cfg(), token_mode=True)
    with pytest.raises(ValueError, match="corpus_identity"):
        check_resume_config(cfg(), cfg(corpus_identity=dict(IDENTITY)), token_mode=True)
