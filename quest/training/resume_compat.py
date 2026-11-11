"""Resume compatibility: what a checkpoint and a resuming run must agree on.

Split out of ``scripts/training/cpt_training_lib`` on 2026-11-04 so the rule can be tested
without importing torch, peft or esm. The trainer imports from here; the names it used to
define locally are kept, so existing callers and tests are unaffected.

Why the objective is checked at all. `check_resume_config` compared 26 fields and
`loss_weighting` was not among them, while the resume path restores optimizer and scheduler
state. A row-share checkpoint therefore resumed into a token-share run without a word, taking
its optimizer with it, which is the quiet version of running two objectives in one experiment.

Why the objective carries a VERSION and not just its name. The open decision is whether
`token` means what it means today -- weight each micro-batch by its share of the step's INPUT
LENGTH, while the loss inside the micro-batch is a mean over the realized masked targets -- or
an exact global masked-target mean. Those are different objectives, and the second is the one
people assume from the name "token-uniform", since per-target weight is uniform under the
first only where mask count happens to track length. If the implementation changes under the
same word, a resume must still refuse, so the stored string has to say which meaning was in
force. A new meaning takes a new version and never reuses an old one.
"""
import json
from typing import Optional

# Config fields that fix the data order or the optimization; a resume must match the checkpoint
# on all of them. Cadence (eval/save/logging) and output/wandb names may differ.
# ``rows_per_step`` is checked separately so a token-budget run can resume on a different GPU
# count (the sampler is world-size invariant).
RESUME_MATCH_KEYS = (
    "base_model", "dataset_path", "canonical_eval_path", "lora_r", "lora_alpha", "lora_dropout",
    "lora_target_modules", "learning_rate", "weight_decay", "adam_beta1", "adam_beta2",
    "warmup_ratio", "gradient_clip", "num_epochs", "max_length", "mlm_probability",
    "eos_strategy", "token_budget", "bucket_boundaries", "use_length_bucketing",
    "region_eval", "tcr_variant", "mhc_variant", "early_stop_metric_path",
    "early_stop_n_patience", "early_stop_min_rel", "seed",
)
# The Stage 4.1 sampler stride-shards per rank, so its data order also depends on these.
RESUME_MATCH_KEYS_LEGACY = ("batch_size", "gradient_accumulation_steps", "world_size")

# What each objective name MEANS in this revision. Changing an implementation means adding a
# version here, never editing one in place: an edited string would silently re-label every
# checkpoint already on disk.
LOSS_WEIGHTING_SEMANTICS = {
    # Every row in the optimizer step contributes equally; long rows are diluted per token.
    "row": "row-share/v1",
    # Micro-batch weight is its share of the step's summed INPUT LENGTH, applied to a loss that
    # is already a mean over that micro-batch's realized masked targets. Per supervised target
    # this is uniform only where mask count is proportional to length. This is what Stage 4.1 and
    # 4.3 ran, so the label stays v1 and the behaviour stays reachable under this name.
    "token": "input-length-share/v1",
    # The objective approved on 2026-11-04: summed loss over every supervised target in the
    # optimizer step, divided by their count across all ranks. A DIFFERENT name rather than a
    # new version of `token`, so that a name never describes behaviour the loop does not have.
    # Implemented in quest/training/loss_normalization.py (exact_step_batches); the loop
    # buffers each optimizer step to learn its global target count (landed 2026-11-11).
    "token_exact": "global-masked-target-mean/v2",
}
UNVERSIONED = "UNVERSIONED"


def objective_fingerprint(d: dict) -> Optional[str]:
    """``<weighting>:<semantics>`` for a config dict, or None if it names no objective.

    A checkpoint written before the semantics were recorded reports ``UNVERSIONED``: it states
    a word without its meaning, which is not the same as agreeing with today's meaning.
    """
    w = d.get("loss_weighting")
    if w is None:
        return None
    return f"{w}:{d.get('loss_weighting_semantics') or UNVERSIONED}"


def check_resume_config(saved: dict, current: dict, *, token_mode: bool,
                        assume_saved_objective: Optional[str] = None) -> None:
    """Raise ``ValueError`` naming every field on which the checkpoint and the current run
    disagree. Values are compared through JSON so tuple/list spellings are equal.

    The objective is checked separately from the generic fields so its refusal can say what
    changed and what the one explicit route past it is. ``assume_saved_objective`` is that
    route: the operator states what they believe an unversioned checkpoint was trained under,
    and it is verified against the checkpoint rather than trusted. It can resolve missing
    metadata; it can never license an actual change of objective.
    """
    saved_fp, current_fp = objective_fingerprint(saved), objective_fingerprint(current)

    if assume_saved_objective is not None:
        if saved_fp is not None and saved_fp != UNVERSIONED and not _assumption_fits(saved, assume_saved_objective):
            raise ValueError(
                f"assume_saved_objective={assume_saved_objective!r} does not match the checkpoint, "
                f"which records {saved_fp!r}. Correct the assumption rather than overriding it."
            )
        saved_fp = assume_saved_objective

    if saved_fp is None:
        raise ValueError(
            "this checkpoint records no training objective (no 'loss_weighting'), so resuming "
            "would adopt the current one by default and mix two objectives in one run. State it "
            "explicitly with assume_saved_objective='<weighting>:<semantics>' after confirming "
            f"what the checkpoint was trained under. Known semantics: {LOSS_WEIGHTING_SEMANTICS}"
        )
    if saved_fp != current_fp:
        raise ValueError(
            f"resume objective mismatch: checkpoint={saved_fp!r} now={current_fp!r}. Optimizer "
            "and scheduler state carry over on resume, so the run would train one objective on "
            "another's optimizer. Resume under the checkpoint's objective, or start a new run. "
            "If only the recorded semantics are missing, confirm them and pass "
            "assume_saved_objective."
        )

    keys = RESUME_MATCH_KEYS + ("rows_per_step",) + (() if token_mode else RESUME_MATCH_KEYS_LEGACY)
    norm = lambda v: json.dumps(v, sort_keys=True, default=str)  # noqa: E731
    bad = [k for k in keys if norm(saved.get(k)) != norm(current.get(k))]
    # A step cap sets the schedule's length. Checkpoints written before it existed have no
    # such field and ran whole epochs, which is what 0 means.
    if (saved.get("max_steps") or 0) != (current.get("max_steps") or 0):
        bad.append("max_steps")
    if bad:
        detail = ", ".join(f"{k}: checkpoint={saved.get(k)!r} now={current.get(k)!r}" for k in bad)
        raise ValueError(f"resume config mismatch on {len(bad)} field(s) — {detail}")


def _assumption_fits(saved: dict, assumption: str) -> bool:
    """True when the assumption agrees with whatever the checkpoint does record.

    An unversioned checkpoint states its weighting but not its meaning, so the assumption has
    to agree on the weighting and may supply the meaning.
    """
    want_w, _, _want_sem = assumption.partition(":")
    if saved.get("loss_weighting") != want_w:
        return False
    have_sem = saved.get("loss_weighting_semantics")
    return have_sem is None or have_sem == _want_sem
