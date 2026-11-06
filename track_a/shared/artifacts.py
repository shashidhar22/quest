"""Digests of the artifacts an evaluation actually read, and the one rule for finding them.

Audit finding 1, 2026-11-06: provenance was a set of launcher DECLARATIONS. The launcher hashed
`$POOL_FILE`, while the evaluator read `load_candidate_pool()`, which resolves its own directory
from `QUEST_SPLITS_DIR` or a fallback chain. The two could name different files, and the manifest
would still record a hash -- of a file nothing read. Split and checkpoint contents were not
recorded at all.

So the resolution rule lives HERE, once, and `load_splits` delegates to it: a hash computed
through this module is a hash of the file the loader would open. The evaluator records its own
digests into its output, and the launcher and comparator validate THOSE rather than re-deriving
anything. Provenance then describes what happened instead of what was intended.

Stdlib only: the launcher's `provenance.py`, the comparator and the judge all import it, and two
of the three must run where pandas and torch are absent.
"""
import hashlib
import json
import os
import pathlib
import subprocess

DIGEST_CHARS = 16          # 64 bits of sha256, printed; collisions are not the threat model here
POOL_BASENAME = "as_class_i_candidate_pool.tsv"

# How a recorded CPT config names its objective. These must agree with
# `quest.training.resume_compat.LOSS_WEIGHTING_SEMANTICS`, which is what a RESUME refuses on;
# a manifest and a resume disagreeing about the same checkpoint would be worse than neither.
# tests/test_artifact_provenance.py checks the two tables against each other.
TOKEN_BUDGET_SEMANTICS = {
    "row": "row-share/v1",
    "token": "input-length-share/v1",
    "token_exact": "global-masked-target-mean/v2",
}
# The Stage 4.1 path, which that table has no row for because it is not a WEIGHTING at all: with
# no token budget every micro-batch is a fixed 16 rows and carries 1/n_acc, so the objective is
# the mean of equally weighted micro-batch means. Arm D ran this. Calling it row share, as the
# judge-first manifests did, names an objective it never trained under.
FIXED_BATCH_SEMANTICS = "fixed-batch-mean/v0"
# Where an adapter directory may record what it was, most trustworthy first.
LINEAGE_FILES = ("trainer_state.json", "training_args.json", "cpt_run_config.json", "args.json")


def splits_dir() -> pathlib.Path:
    """Where benchmark_v2's AS splits live. The single rule; `load_splits` calls this.

    Resolved inside this checkout first so the judge is runnable and testable off the training
    box; the box layout stays as a fallback. Override with QUEST_SPLITS_DIR. If none exists the
    first candidate is returned, so a loader's FileNotFoundError names a path the caller can act
    on rather than one that was never going to exist.
    """
    env = os.environ.get("QUEST_SPLITS_DIR")
    candidates = (
        *((pathlib.Path(env),) if env else ()),
        pathlib.Path(__file__).resolve().parents[2] / "benchmark_v2" / "splits",
        pathlib.Path("/home/ubuntu/quest/benchmark_v2/splits"),
    )
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def file_digest(path) -> str:
    """sha256 prefix of one file's bytes, or 'MISSING'."""
    p = pathlib.Path(path)
    if not p.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:DIGEST_CHARS]


def dir_digest(path) -> str:
    """Digest of a directory's contents: every file's relative path, size and bytes.

    Used for an adapter directory, where the weights are one file among several that decide
    what the adapter IS (`adapter_config.json` carries the LoRA shape and target modules).
    Paths are included, so a renamed file changes the digest.
    """
    p = pathlib.Path(path)
    if not p.is_dir():
        return "MISSING"
    h = hashlib.sha256()
    for f in sorted(q for q in p.rglob("*") if q.is_file()):
        h.update(str(f.relative_to(p)).encode())
        h.update(str(f.stat().st_size).encode())
        h.update(file_digest(f).encode())
    return h.hexdigest()[:DIGEST_CHARS]


def pool_path() -> pathlib.Path:
    return splits_dir() / POOL_BASENAME


def split_path(task: str, condition: str) -> pathlib.Path:
    return splits_dir() / f"{task}_{condition}.parquet"


def split_digests(task: str, conditions) -> dict:
    """{condition: digest} for the split files a run reads. 'MISSING' is recorded, not raised:
    a preflight decides what to do about it, and a record of absence is still a record."""
    return {c: file_digest(split_path(task, c)) for c in conditions}


def code_commit(repo_root=None) -> str:
    """The commit the code being run came from, or 'unknown'/'dirty:<sha>'.

    The audit's point about deployment: local tracking refs said 9 commits ahead, which is not
    evidence about what is installed on the box. A result should carry the revision that produced
    it, read from the working tree that produced it.
    """
    root = pathlib.Path(repo_root or pathlib.Path(__file__).resolve().parents[2])
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return "unknown"
        head = sha.stdout.strip()[:12]
        # TRACKED files only. Hashing the untracked listing made the identity change as a run
        # wrote its own outputs inside the checkout, so the base and arm-D evaluations of ONE run
        # recorded different code and the comparison refused. Untracked files are not code git
        # can vouch for either way; the box's untracked tree is data, logs and preserved V03
        # scripts that nothing here executes.
        st = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                             "--untracked-files=no"],
                            capture_output=True, text=True, timeout=120)
        if st.returncode != 0:
            return "unknown"
        if not st.stdout.strip():
            return head
        # `dirty:<sha>` alone made any two dirty trees on one HEAD compare equal, so code edited
        # on the box between two evaluations was certified as the same code (pre-launch review,
        # 2026-11-06). The tracked diff is hashed into the identity.
        diff = subprocess.run(["git", "-C", str(root), "diff", "HEAD", "--binary"],
                              capture_output=True, timeout=120)
        if diff.returncode != 0:
            return "unknown"
        return f"dirty:{head}:{hashlib.sha256(diff.stdout).hexdigest()[:12]}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _recorded_config(adapter_dir):
    """(config dict, global_step, source filename) from whatever the adapter recorded, or Nones.

    `trainer_state.json` nests the run's config under "config"; a flat file is also accepted.
    Nothing is inferred from a filename or a path: an unreadable or silent artifact stays silent.
    """
    for name in LINEAGE_FILES:
        f = pathlib.Path(adapter_dir) / name
        if not f.is_file():
            continue
        try:
            d = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(d, dict):
            continue
        cfg = d.get("config") if isinstance(d.get("config"), dict) else d
        step = d.get("global_step") if isinstance(d.get("global_step"), int) else cfg.get("global_step")
        return cfg, step, name
    return None, None, None


INFERRED_FIXED_BATCH = ("inferred: the recorded config has a fixed batch_size and "
                        "gradient_accumulation_steps and no token_budget field, i.e. it predates "
                        "token-budget batching, whose only path was the fixed-batch one")


def objective_basis(cfg) -> str:
    """How `objective_from_config` knows what it says: `recorded`, `inferred: ...`, or
    `unrecorded`. Stated beside the name, because arm D's objective is an inference from a
    complete config dump that has no token_budget field -- sound, but not a reading."""
    if not isinstance(cfg, dict):
        return "unrecorded"
    if "token_budget" in cfg:
        return "recorded"
    if cfg.get("batch_size") and cfg.get("gradient_accumulation_steps"):
        return INFERRED_FIXED_BATCH
    return "unrecorded"


def objective_from_config(cfg) -> str:
    """The objective a recorded CPT config implies, by the rule the trainer actually follows.

    `micro_batch_loss_scale` branches on token_mode FIRST: with no token budget the weighting
    name is not read at all, so a config carrying `loss_weighting: row` and `token_budget: 0`
    trained the fixed-batch objective, not row share. Deriving the name from the weighting alone
    is how arm D came to be labelled with an objective it never used.
    """
    if not isinstance(cfg, dict) or objective_basis(cfg) == "unrecorded":
        # Neither a token_budget field nor the fixed-batch shape: absence is evidence of nothing.
        return "unrecorded"
    budget = cfg.get("token_budget") or 0
    try:
        budget = int(budget)
    except (TypeError, ValueError):
        return "unrecorded"
    if budget <= 0:
        return FIXED_BATCH_SEMANTICS
    weighting = cfg.get("loss_weighting")
    return TOKEN_BUDGET_SEMANTICS.get(weighting, f"unknown-weighting:{weighting!r}")


def cpt_lineage(adapter) -> dict:
    """What an adapter says about its own CPT history, read from the adapter directory.

    Deliberately NOT a launcher declaration. Both judge-first manifests were labelled
    `objective=legacy_row_share_cpt`: wrong for the base arm, which has no CPT at all, and wrong
    for arm D as well, whose own `trainer_state.json` records a fixed batch of 16 with
    accumulation 16 and no token budget. A label applied by the script that happens to be
    running is not lineage. So this reports what the artifact records, `unrecorded` when the
    artifact is silent, and `none` when there is no adapter -- and never a guess.
    """
    if adapter in (None, "", "none", "base"):
        return {"cpt_adapter": "none", "cpt_adapter_digest": "none", "cpt_objective": "none",
                "cpt_run_id": "none", "cpt_steps": None, "cpt_seed": None,
                "cpt_token_budget": None, "cpt_lineage_source": "no adapter",
                "cpt_objective_basis": "no adapter"}
    p = pathlib.Path(adapter)
    cfg, step, src = _recorded_config(p)
    budget = (cfg or {}).get("token_budget")
    return {
        "cpt_adapter": str(p),
        "cpt_adapter_digest": dir_digest(p),
        # the run id is the directory a human can look up; `final/` is a leaf, not an id
        "cpt_run_id": p.parent.name if p.name in ("final", "best", "last") else p.name,
        "cpt_objective": objective_from_config(cfg) if cfg is not None else "unrecorded",
        "cpt_objective_basis": objective_basis(cfg),
        "cpt_steps": step,
        "cpt_seed": (cfg or {}).get("seed"),
        "cpt_token_budget": 0 if cfg is not None and not budget else budget,
        "cpt_lineage_source": src or "the adapter records nothing",
    }
