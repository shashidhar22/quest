#!/usr/bin/env python3
"""The step-12 pilot's experiment lock: every open value frozen in one file, and the only source
of the arguments the pilot driver passes to the trainer and the judge.

new-cpt-plan (2026-11-10) asks that every open value be resolved in a single
`experiment-lock.json` before anything is spent. A lock is only worth having if the run cannot
drift from it, so this module is used at both ends:

    write       builds the lock from artifacts on disk (markers, split files, the code itself)
                plus the frozen constants below. Declared identities are never accepted.
    verify      rebuilds the same thing from the same inputs and lists EVERY difference.
    cpt-args    the trainer's argv, derived from the lock alone (the driver hard-codes nothing).
    sft-args    the judge's argv for train and eval, likewise.
    check-cpt   a CPT run's final/trainer_state.json against the lock.
    check-pair  the two CPT arms against each other: one initial LoRA state, one target budget.
    check-eval  an SFT cell's eval.json against the lock.

Every refusal exits non-zero with the full list of reasons; absence never counts as agreement,
so a field missing on both sides is a mismatch. Stdlib only at import, because the driver calls
this many times on a box whose torch import costs seconds; the SFT recipe is read from
`track_a/esm_sft/scoring.py`, which needs NumPy and nothing heavier.

Code is pinned by file digest rather than a commit id because the lock is committed in the same
commit as the code it pins. The pinned set is computed, not listed: every repo module the
trainer, the judge and the read-out import, transitively (`code_files`). A hand-kept list left
out the samplers, the collators and the negative sampler, which decide the experiment as much
as the files that import them.
"""
from __future__ import annotations

import argparse
import array
import ast
import datetime
import importlib.util
import json
import pathlib
import math
import re
import struct
import sys

QUEST = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(QUEST))
from quest.data.corpus_preflight import MARKER, sha256_file  # noqa: E402  (stdlib only)
from track_a.shared.artifacts import POOL_BASENAME, dir_digest, file_digest  # noqa: E402  (stdlib only)

SCHEMA = "experiment-lock/v1"
# What the pilot executes: the trainer, the judge, the read-out and this module. Everything
# they import from the repo is pinned with them; the driver is pinned as a file.
ENTRY_POINTS = (
    "scripts/training/train_cpt_stage2_single.py",
    "track_a/esm_sft/judge.py",
    "scripts/training/pilot12_box/pilot_compare.py",
    "scripts/training/pilot12_box/experiment_lock.py",
)
PINNED_FILES = ("scripts/training/pilot12_box/run_pilot.sh",)
# Directories the scripts put on sys.path, where a bare `import name` may resolve.
SCRIPT_DIRS = ("scripts/training", "scripts/training/stage42_box", "scripts/training/v04_box",
               "scripts/training/pilot12_box")


def code_files(repo) -> list:
    """Repo-relative paths of ENTRY_POINTS, every repo module they import (transitively, package
    `__init__` files included, imports inside functions included) and PINNED_FILES. Imports that
    do not resolve inside the repo are third-party and are not pinned here."""
    root = pathlib.Path(repo).resolve()

    def resolve(module, importer):
        parts = module.split(".")
        for base in (root, importer.parent, *(root / d for d in SCRIPT_DIRS)):
            for cand in (base.joinpath(*parts).with_suffix(".py"), base.joinpath(*parts, "__init__.py")):
                if cand.is_file():
                    return cand
        return None

    seen, todo = set(), [root / e for e in ENTRY_POINTS if (root / e).is_file()]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
            elif isinstance(node, ast.ImportFrom):                      # relative import
                base = f.parent
                for _ in range(node.level - 1):
                    base = base.parent
                pkg = base.joinpath(*(node.module or "").split(".")) if node.module else base
                todo += [c for a in node.names
                         for c in (pkg.with_suffix(".py"), pkg / "__init__.py", (pkg / a.name).with_suffix(".py"))
                         if c.is_file()]
                continue
            else:
                continue
            for name in names:
                parts = name.split(".")
                todo += [p for p in (root.joinpath(*parts[:i], "__init__.py") for i in range(1, len(parts)))
                         if p.is_file()]
                hit = resolve(name, f)
                if hit is not None:
                    todo.append(hit)
    return sorted({str(p.relative_to(root)) for p in seen} | set(PINNED_FILES))
# Which published corpus each CPT arm trains on. A swapped pair is refused at write and at
# cpt-args, because both corpora pass the same preflight and nothing else would notice.
ARM_CORPUS = {"P": "repaired_proportional_10M", "E": "task_enriched_cap1000"}
CORPUS_IDENTITY = ("run_id", "contract_sha256", "marker_sha256")   # what the trainer records
TASK = "as_paired_i"
BENCHMARK_CONDITIONS = ("train", "val", "test_iid", "test_novel_pep", "test_level4")
SFT_SPLITS = ("train", "val")
HEX64 = re.compile(r"[0-9a-f]{64}")

# The frozen values (plans/step12-diagnostic-and-pilot.md, nodded 2026-11-11). Everything not
# here is read from an artifact at write time.
FROZEN = {
    "stage": "pilot-10pct",
    "question": ("does CPT on a validated repaired corpus improve downstream class-I peptide "
                 "retrieval over a fresh, equally capable SFT-only base, and does the enriched "
                 "sampling package improve over repaired proportional sampling?"),
    "primary_contrast": "E-B",
    "secondary_contrasts": ["P-B", "E-P"],
    "replication": "one CPT seed, one SFT seed: exploratory, cannot establish superiority",
    "model": {"base_model": "esmc_300m", "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05},
    "cpt": {
        "loss_weighting": "token_exact",
        "loss_weighting_semantics": "global-masked-target-mean/v2",
        "token_budget": 8192, "batch_size": 16,
        "gradient_accumulation_steps": {"P": 64, "E": 47},
        "max_steps": 780, "num_epochs": 1, "lr": 2e-4, "warmup_ratio": 0.05,
        "weight_decay": 0.01, "gradient_clip": 1.0, "mlm_probability": 0.15,
        "max_length": 1024, "bucket_boundaries": [64, 128, 256, 512, 1024],
        "eos_strategy": "raw", "tcr_variant": "cdr3", "mhc_variant": "full", "seed": 42,
        "early_stop_n_patience": 0, "checkpoint": "final",
        "full_target_budget": 23_000_000, "pilot_target_budget": 2_300_000,
        "arm_target_tolerance": 0.02,
        "basis": ("Measured on visit S12 at 1,024 rows per step: P 2,951 and E 4,002 masked "
                  "targets per step; E's accumulation of 47 gives 752 rows per step, so both arms "
                  "take about 2,950 targets per step for the same 780 updates."),
    },
    "sft": {"task": TASK, "seed": 42, "selector": "best.pt by validation AUROC"},
    "evaluation": {
        "conditions": ["val"], "distractor_draw": "length_matched", "max_queries": 5000,
        "min_peptides": 100,
        "comparator": {"script": "scripts/training/pilot12_box/pilot_compare.py",
                       "metrics": ["norm_rank", "recall_at_10", "mrr"],
                       "units": ["peptide", "pep_cluster"], "n_boot": 4000, "seed": 42},
        "gate3": "not done: validation only with the current evaluator; every test part stays closed",
    },
    "spending": {"usd_per_hour": 1.069, "cpt_stage_max_hours": 2, "visit_max_hours": 24,
                 "visit_cap_usd": 26},
    "limitations": [
        "masking has no dedicated RNG stream",
        "SFT hours taken from V04a, not re-measured",
        "validation evaluation never timed",
        "native precision",
        "base model weights and tokenizer are not pinned by digest; third-party packages are not pinned",
        "the canonical evaluation set's contents are not pinned (it selects nothing: early stop is off)",
        "an SFT checkpoint is bound to its adapter at evaluation time, not at training time",
    ],
}
# judge train flags, in the order printed; `max_len` is shared by train and eval.
SFT_TRAIN_FLAGS = ("max_train_pos", "max_val_pos", "epochs", "patience", "batch_size", "lr",
                   "seed", "num_workers")


class LockError(Exception):
    """Carries every reason at once: a refusal that names one problem hides the rest."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _load_json(path, problems, what):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError) as e:
        problems.append(f"{what}: {path} unreadable ({type(e).__name__})")
        return None


def _marker_identity(version_dir, problems, arm):
    """The corpus identity exactly as `check_corpus_and_objective` returns it, from the marker."""
    path = pathlib.Path(version_dir) / MARKER
    marker = _load_json(path, problems, f"corpora.{arm}") or {}
    contract = marker.get("contract") or {}
    ident = {"version": contract.get("version"), "run_id": marker.get("run_id"),
             "contract_sha256": contract.get("sha256"),
             "marker_sha256": sha256_file(path) if path.is_file() else None}
    for k, v in ident.items():
        if not v:
            problems.append(f"corpora.{arm}.{k}: the marker records none")
    if ident["version"] and ident["version"] != ARM_CORPUS[arm]:
        problems.append(f"corpora.{arm}.version: {ident['version']!r}, but arm {arm} trains on "
                        f"{ARM_CORPUS[arm]!r}")
    return ident


def _sft_defaults(repo, problems):
    """`SFT_TRAIN_DEFAULTS` from the scoring module of the repo being locked or verified."""
    path = pathlib.Path(repo) / "track_a" / "esm_sft" / "scoring.py"
    try:
        spec = importlib.util.spec_from_file_location("_locked_scoring", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        recipe = dict(mod.SFT_TRAIN_DEFAULTS)
    except Exception as e:  # noqa: BLE001  (any failure to read the recipe is a refusal)
        problems.append(f"sft.recipe: cannot read SFT_TRAIN_DEFAULTS from {path} ({type(e).__name__})")
        return None
    unknown = sorted(set(recipe) - set(SFT_TRAIN_FLAGS) - {"max_len"})
    if unknown:
        problems.append(f"sft.recipe: keys no judge flag here expresses: {unknown}")
    if recipe.get("seed") != FROZEN["sft"]["seed"]:
        problems.append(f"sft.recipe: seed {recipe.get('seed')!r} is not the locked SFT seed")
    return recipe


def build(corpora: dict, splits_dir, repo) -> tuple[dict, list]:
    """(lock without created_utc, problems). Problems are collected, never raised, so that
    `verify` can list all of them; `write` refuses on any."""
    problems = []
    if set(corpora) != set(ARM_CORPUS):
        problems.append(f"--corpus must name exactly {sorted(ARM_CORPUS)}, got {sorted(corpora)}")
    repo, splits = pathlib.Path(repo), pathlib.Path(splits_dir)
    code = {}
    for rel in sorted({*code_files(repo), *ENTRY_POINTS}):
        code[rel] = sha256_file(repo / rel) if (repo / rel).is_file() else "MISSING"
        if code[rel] == "MISSING":
            problems.append(f"code_files.{rel}: missing from {repo}")
    bench = {}
    for cond in BENCHMARK_CONDITIONS:
        name = f"{TASK}_{cond}.parquet"
        bench[name] = sha256_file(splits / name) if (splits / name).is_file() else "MISSING"
    digests = {c: file_digest(splits / f"{TASK}_{c}.parquet") for c in SFT_SPLITS}
    pool = file_digest(splits / POOL_BASENAME)
    for k, v in {**{f"benchmark_split_sha256.{n}": d for n, d in bench.items()},
                 **{f"evaluation.split_digests.{c}": d for c, d in digests.items()},
                 "evaluation.pool_digest": pool}.items():
        if v == "MISSING":
            problems.append(f"{k}: file missing in {splits}")
    lock = {
        "schema": SCHEMA,
        **{k: FROZEN[k] for k in ("stage", "question", "primary_contrast", "secondary_contrasts",
                                  "replication")},
        "code_files": code,
        "corpora": {arm: _marker_identity(corpora[arm], problems, arm)
                    for arm in ARM_CORPUS if arm in corpora},
        "benchmark_split_sha256": bench,
        "model": FROZEN["model"],
        "cpt": FROZEN["cpt"],
        "sft": {**FROZEN["sft"], "recipe": _sft_defaults(repo, problems)},
        "evaluation": {**FROZEN["evaluation"], "split_digests": digests, "pool_digest": pool},
        "spending": FROZEN["spending"],
        "limitations": FROZEN["limitations"],
    }
    return lock, problems


def _diff(locked, found, path="") -> list:
    """Every leaf on which two lock documents disagree. A key on one side only is a mismatch."""
    if isinstance(locked, dict) and isinstance(found, dict):
        out = []
        for k in sorted(set(locked) | set(found), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in locked or k not in found:
                out.append(f"{p}: {'absent from the lock' if k not in locked else 'not found now'}")
            else:
                out += _diff(locked[k], found[k], p)
        return out
    return [] if locked == found and locked is not None else [f"{path}: locked {locked!r}, found {found!r}"]


def load_lock(path) -> dict:
    problems = []
    doc = _load_json(path, problems, "lock")
    if doc is not None and doc.get("schema") != SCHEMA:
        problems.append(f"lock: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    if problems:
        raise LockError(problems)
    return doc


# ------------------------------------------------------------------------------ write / verify
def cmd_write(a):
    out = pathlib.Path(a.out)
    if out.exists():
        raise LockError([f"{out} exists; a lock is written once and never overwritten"])
    lock, problems = build(a.corpus, a.splits_dir, a.repo)
    if a.benchmark_sha256:
        given = _load_json(a.benchmark_sha256, problems, "--benchmark-sha256")
        if given is not None and given != lock["benchmark_split_sha256"]:
            problems.append("--benchmark-sha256 is not the split files on disk: "
                            + "; ".join(_diff(given, lock["benchmark_split_sha256"])))
    if problems:
        raise LockError(problems)
    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out, "x") as fh:
        json.dump({"schema": lock["schema"], "created_utc": created,
                   **{k: v for k, v in lock.items() if k != "schema"}}, fh, indent=1)
        fh.write("\n")
    print(f"wrote {out}\nlock sha256: {sha256_file(out)}")


def cmd_verify(a):
    doc = load_lock(a.lock)
    found, problems = build(a.corpus, a.splits_dir, a.repo)
    problems += _diff({k: v for k, v in doc.items() if k != "created_utc"}, found)
    if problems:
        raise LockError(problems)
    print(f"lock verified against the artifacts\nlock sha256: {sha256_file(a.lock)}")


# ------------------------------------------------------------------------------ trainer argv
def trainer_settings(lock, arm) -> list:
    """(trainer flag, CPTConfig field, value) for every setting the lock fixes, in argv order.
    The field names are those `train_cpt_stage2_single.py` builds its CPTConfig with, so the
    same table writes the argv and checks what the run recorded."""
    c, m = lock["cpt"], lock["model"]
    steps = c["max_steps"]
    return [
        ("--base_model", "base_model", m["base_model"]),
        ("--tcr_variant", "tcr_variant", c["tcr_variant"]),
        ("--mhc_variant", "mhc_variant", c["mhc_variant"]),
        ("--eos_strategy", "eos_strategy", c["eos_strategy"]),
        ("--max_length", "max_length", c["max_length"]),
        ("--lora_r", "lora_r", m["lora_r"]),
        ("--lora_alpha", "lora_alpha", m["lora_alpha"]),
        ("--lora_dropout", "lora_dropout", m["lora_dropout"]),
        ("--num_epochs", "num_epochs", c["num_epochs"]),
        ("--batch_size", "batch_size", c["batch_size"]),
        ("--gradient_accumulation_steps", "gradient_accumulation_steps",
         c["gradient_accumulation_steps"][arm]),
        ("--lr", "learning_rate", c["lr"]),
        ("--warmup_ratio", "warmup_ratio", c["warmup_ratio"]),
        ("--weight_decay", "weight_decay", c["weight_decay"]),
        ("--mlm_probability", "mlm_probability", c["mlm_probability"]),
        ("--gradient_clip", "gradient_clip", c["gradient_clip"]),
        # One checkpoint, at the end. The cadence is one step past the cap so the loop never
        # evaluates: the trainer's final evaluation follows at once, and a cadence equal to the
        # cap would run the same canonical evaluation twice.
        ("--eval_steps", "eval_every_n_steps", steps + 1),
        ("--save_steps", "save_every_n_steps", steps),
        ("--bucket_boundaries", "bucket_boundaries", c["bucket_boundaries"]),
        ("--no_compile", "use_compile", False),
        ("--token_budget", "token_budget", c["token_budget"]),
        ("--max_steps", "max_steps", steps),
        ("--corpus_preflight", "corpus_preflight", "required"),
        ("--loss_weighting", "loss_weighting", c["loss_weighting"]),
        ("--early_stop_n_patience", "early_stop_n_patience", c["early_stop_n_patience"]),
        ("--report_to", "report_to", "none"),
        ("--seed", "seed", c["seed"]),
    ]


def _argv_items(flag, value):
    if value is False:                 # the one store_true flag: --no_compile means use_compile False
        return [flag]
    if isinstance(value, list):
        return [flag, *map(str, value)]
    return [flag, str(value)]


def cmd_cpt_args(a):
    lock = load_lock(a.lock)
    dataset = pathlib.Path(a.dataset_path)
    problems = []
    if dataset.name != "dataset":
        problems.append(f"--dataset-path must be <version_dir>/dataset, not {dataset}")
    ident = _marker_identity(dataset.parent, problems, a.arm)
    for k in CORPUS_IDENTITY:
        if ident.get(k) != lock["corpora"][a.arm].get(k):
            problems.append(f"corpora.{a.arm}.{k}: {dataset.parent} is not the locked corpus")
    # The trainer's check that the corpus's leakage audit read THESE benchmark files is always
    # on: the file it is given is the lock's own table, written here on first use, and a file
    # already there must say the same.
    bench = pathlib.Path(a.benchmark_sha256)
    if bench.exists():
        given = _load_json(bench, problems, "--benchmark-sha256")
        if given is not None and given != lock["benchmark_split_sha256"]:
            problems.append("--benchmark-sha256 is not the locked benchmark_split_sha256")
    if problems:
        raise LockError(problems)
    if not bench.exists():
        with open(bench, "x") as fh:
            json.dump(lock["benchmark_split_sha256"], fh, indent=1)
            fh.write("\n")
    argv = ["--dataset_path", str(dataset), "--canonical_eval_path", a.canonical_eval_path,
            "--output_dir", a.output_dir]
    for flag, _field, value in trainer_settings(lock, a.arm):
        argv += _argv_items(flag, value)
    argv += ["--corpus_benchmark_sha256", str(bench)]
    print("\n".join(argv))


def cmd_sft_args(a):
    lock = load_lock(a.lock)
    sft, ev, c = lock["sft"], lock["evaluation"], lock["cpt"]
    recipe = sft.get("recipe")
    if not isinstance(recipe, dict) or set(recipe) != set(SFT_TRAIN_FLAGS) | {"max_len"}:
        raise LockError([f"sft.recipe in the lock is not a complete recipe: {recipe!r}"])
    need = {"train": ("out_dir",), "eval": ("ckpt", "out")}[a.stage]
    missing = [f"--{k.replace('_', '-')}" for k in need if not getattr(a, k)]
    if missing:
        raise LockError([f"sft-args --stage {a.stage} needs {' '.join(missing)}"])
    argv = [a.stage, "--task", sft["task"], "--adapter", a.adapter,
            "--tcr_variant", c["tcr_variant"], "--mhc_variant", c["mhc_variant"],
            "--max_len", str(recipe["max_len"])]
    if a.stage == "train":
        argv += ["--out_dir", a.out_dir]
        for k in SFT_TRAIN_FLAGS:
            argv += [f"--{k}", str(recipe[k])]
    else:
        argv += ["--ckpt", a.ckpt, "--out", a.out, "--conditions", *ev["conditions"],
                 "--max_queries", str(ev["max_queries"]), "--min_peptides", str(ev["min_peptides"]),
                 "--distractor_draw", ev["distractor_draw"]]
    print("\n".join(argv))


# ------------------------------------------------------------------------------ checks
def _same(name, locked, found, problems):
    if locked is None or found is None or locked != found:
        problems.append(f"{name}: locked {locked!r}, recorded {found!r}")


def _record(path, problems, what) -> dict:
    """A JSON object with something in it, or {} and a problem. An empty or non-object file is
    a record of nothing, and nothing must never read as "no problems"."""
    doc = _load_json(path, problems, what)
    if isinstance(doc, dict) and doc:
        return doc
    if doc is not None or not problems:
        problems.append(f"{what}: {path} is empty or not a JSON object")
    return {}


def _state(final, problems):
    return _record(pathlib.Path(final) / "trainer_state.json", problems, "trainer_state")


def adapter_problems(final) -> list:
    """The adapter an SFT cell will be built on is complete and finite. The trainer logs a
    failed save and still writes its state file, and a diverged run saves NaN weights without
    complaint; either would cost three SFT cells before anything else noticed."""
    final = pathlib.Path(final)
    problems = [] if (final / "adapter_config.json").is_file() else [f"adapter_config.json: missing in {final}"]
    return problems + _weights_problems(final / "adapter_model.safetensors")


def _weights_problems(path) -> list:
    final = path.parent
    try:
        raw = path.read_bytes()
        (n,) = struct.unpack("<Q", raw[:8])
        header = json.loads(raw[8:8 + n])
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        data = raw[8 + n:]
        if not tensors or max(v["data_offsets"][1] for v in tensors.values()) != len(data):
            return [f"adapter_model.safetensors: no tensors, or truncated, in {final}"]
        for name, v in tensors.items():
            if v["dtype"] == "F32":
                values = array.array("f", data[v["data_offsets"][0]:v["data_offsets"][1]])
                if sys.byteorder != "little":
                    values.byteswap()
                if not math.isfinite(math.fsum(values)):
                    return [f"adapter_model.safetensors: non-finite values in {name}"]
    except (OSError, ValueError, KeyError, TypeError, struct.error) as e:
        return [f"adapter_model.safetensors: unreadable or not a safetensors file in {final} ({type(e).__name__})"]
    return []


def cpt_problems(lock, arm, final) -> list:
    problems = []
    st = _state(final, problems)
    if not st:
        return problems
    c = lock["cpt"]
    _same("global_step", c["max_steps"], st.get("global_step"), problems)
    _same("loss_weighting_semantics", c["loss_weighting_semantics"],
          st.get("loss_weighting_semantics"), problems)
    ident = st.get("corpus_identity") if isinstance(st.get("corpus_identity"), dict) else {}
    for k in CORPUS_IDENTITY:
        _same(f"corpus_identity.{k}", lock["corpora"][arm].get(k), ident.get(k), problems)
    init = st.get("initial_lora_sha256")
    if not (isinstance(init, str) and HEX64.fullmatch(init)):
        problems.append(f"initial_lora_sha256: {init!r} is not a sha256")
    seen = st.get("masked_targets_seen")
    if not (isinstance(seen, int) and not isinstance(seen, bool) and seen > 0):
        problems.append(f"masked_targets_seen: {seen!r}; a token_exact run counts its targets")
    problems += adapter_problems(final)
    cfg = st.get("config") if isinstance(st.get("config"), dict) else {}
    for _flag, field, value in trainer_settings(lock, arm):
        if field not in cfg:
            problems.append(f"config.{field}: absent from the recorded config")
        else:
            _same(f"config.{field}", value, cfg[field], problems)
    return problems


def cmd_check_cpt(a):
    problems = cpt_problems(load_lock(a.lock), a.arm, a.final)
    if problems:
        raise LockError(problems)
    print(f"cpt {a.arm}: {a.final} matches the lock")


def cmd_check_pair(a):
    lock = load_lock(a.lock)
    c, problems = lock["cpt"], []
    states = {"P": _state(a.final_p, problems), "E": _state(a.final_e, problems)}
    inits = {k: s.get("initial_lora_sha256") for k, s in states.items()}
    for k, v in inits.items():
        if not (isinstance(v, str) and HEX64.fullmatch(v)):
            problems.append(f"{k}: initial_lora_sha256 {v!r} is not a sha256")
    if inits["P"] != inits["E"]:
        problems.append(f"initial LoRA states differ: P {inits['P']!r}, E {inits['E']!r}")
    seen = {k: s.get("masked_targets_seen") for k, s in states.items()}
    tol, budget = c["arm_target_tolerance"], c["pilot_target_budget"]
    if all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in seen.values()):
        gap = abs(seen["P"] - seen["E"]) / max(seen.values())
        print(f"masked targets: P {seen['P']}, E {seen['E']}; relative gap {gap:.4f} "
              f"(tolerance {tol}); pilot budget {budget}")
        if gap > tol:
            problems.append(f"arms {gap:.4f} apart in masked targets, above {tol}")
        for k, v in seen.items():
            off = abs(v - budget) / budget
            if off > tol:
                problems.append(f"{k}: {v} masked targets is {off:.4f} off the budget {budget}")
    else:
        problems.append(f"masked_targets_seen not recorded as positive counts: {seen}")
    if problems:
        raise LockError(problems)
    print("pair: one initial LoRA state, targets within tolerance")


def eval_problems(lock, doc, arm, final=None) -> list:
    from track_a.esm_sft.scoring import sft_recipe   # NumPy only; the judge's own recipe string
    problems = []
    ev, sft, c = lock["evaluation"], lock["sft"], lock["cpt"]
    conds = sorted((doc.get("conditions") or {}).keys())
    if conds != sorted(ev["conditions"]):
        problems.append(f"conditions: scored {conds}, the lock allows {ev['conditions']}")
    args, prov = doc.get("args") or {}, doc.get("provenance") or {}
    # A smoke evaluation scores 20 queries and looks like any other file. Not recorded is not "no".
    if args.get("smoke") is not False:
        problems.append(f"args.smoke: {args.get('smoke')!r}; the pilot takes no smoke evaluation")
    _same("args.task", sft["task"], args.get("task"), problems)
    for k in ("distractor_draw", "max_queries", "min_peptides"):
        _same(f"args.{k}", ev[k], args.get(k), problems)
    _same("provenance.split_digests", {k: ev["split_digests"].get(k) for k in ev["conditions"]},
          prov.get("split_digests"), problems)
    _same("provenance.pool_digest", ev["pool_digest"], prov.get("pool_digest"), problems)
    _same("provenance.sft_train_digests", ev["split_digests"], prov.get("sft_train_digests"), problems)
    _same("provenance.sft_seed", sft["seed"], prov.get("sft_seed"), problems)
    _same("provenance.sft_max_len", sft["recipe"]["max_len"], prov.get("sft_max_len"), problems)
    _same("provenance.sft_recipe", sft_recipe(sft["recipe"]), prov.get("sft_recipe"), problems)
    if arm == "B":
        _same("provenance.cpt_adapter", "none", prov.get("cpt_adapter"), problems)
    else:
        if prov.get("cpt_adapter") in (None, "", "none"):
            problems.append(f"provenance.cpt_adapter: arm {arm} was evaluated with no CPT adapter")
        _same("provenance.cpt_objective", c["loss_weighting_semantics"], prov.get("cpt_objective"), problems)
        _same("provenance.cpt_steps", c["max_steps"], prov.get("cpt_steps"), problems)
        _same("provenance.cpt_seed", c["seed"], prov.get("cpt_seed"), problems)
        _same("provenance.cpt_token_budget", c["token_budget"], prov.get("cpt_token_budget"), problems)
        # Both arms record the same objective, steps, seed and budget, so only the adapter's
        # digest says WHICH arm this cell was built on. `final` is the directory check-cpt
        # held to this arm's corpus.
        found = dir_digest(final) if final else None
        _same("provenance.cpt_adapter_digest", None if found == "MISSING" else found,
              prov.get("cpt_adapter_digest"), problems)
    return problems


def cmd_check_eval(a):
    problems = []
    doc = _record(a.eval, problems, "eval")
    if doc:
        problems = eval_problems(load_lock(a.lock), doc, a.arm, a.final)
    if problems:
        raise LockError(problems)
    print(f"eval {a.arm}: {a.eval} matches the lock")


def cmd_adapter_digest(a):
    """The digest the judge records for an adapter directory, so the driver can hold both arms'
    adapters to what they were when the pair was checked."""
    digest = dir_digest(a.final)
    if digest == "MISSING":
        raise LockError([f"{a.final} is not a directory"])
    print(digest)


# ------------------------------------------------------------------------------ CLI
def _corpora(items):
    out = {}
    for item in items or ():
        arm, sep, path = item.partition("=")
        if not sep or arm not in ARM_CORPUS or arm in out:
            raise LockError([f"--corpus takes P=<dir> and E=<dir> once each, not {item!r}"])
        out[arm] = path
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def inputs(p):
        p.add_argument("--corpus", action="append", required=True, metavar="ARM=VERSION_DIR")
        p.add_argument("--splits-dir", required=True)
        p.add_argument("--repo", required=True)

    w = sp.add_parser("write")
    w.add_argument("--out", required=True)
    inputs(w)
    w.add_argument("--benchmark-sha256", default=None)
    v = sp.add_parser("verify")
    v.add_argument("--lock", required=True)
    inputs(v)
    ca = sp.add_parser("cpt-args")
    ca.add_argument("--lock", required=True)
    ca.add_argument("--arm", required=True, choices=sorted(ARM_CORPUS))
    ca.add_argument("--dataset-path", required=True)
    ca.add_argument("--output-dir", required=True)
    ca.add_argument("--canonical-eval-path", required=True)
    ca.add_argument("--benchmark-sha256", required=True,
                    help="where the lock's benchmark hashes are (or will be written) for the trainer")
    sa = sp.add_parser("sft-args")
    sa.add_argument("--lock", required=True)
    sa.add_argument("--stage", required=True, choices=["train", "eval"])
    sa.add_argument("--adapter", required=True)
    sa.add_argument("--out-dir", dest="out_dir", default=None)
    sa.add_argument("--ckpt", default=None)
    sa.add_argument("--out", default=None)
    cc = sp.add_parser("check-cpt")
    cc.add_argument("--lock", required=True)
    cc.add_argument("--arm", required=True, choices=sorted(ARM_CORPUS))
    cc.add_argument("--final", required=True)
    cp = sp.add_parser("check-pair")
    cp.add_argument("--lock", required=True)
    cp.add_argument("--final-p", required=True)
    cp.add_argument("--final-e", required=True)
    ce = sp.add_parser("check-eval")
    ce.add_argument("--lock", required=True)
    ce.add_argument("--eval", required=True)
    ce.add_argument("--arm", required=True, choices=["B", *sorted(ARM_CORPUS)])
    ce.add_argument("--final", default=None, help="the arm's CPT final/ directory; needed for P and E")
    ad = sp.add_parser("adapter-digest")
    ad.add_argument("--lock", default=None, help="accepted and unused, so the driver calls every subcommand alike")
    ad.add_argument("--final", required=True)
    a = ap.parse_args(argv)
    try:
        if hasattr(a, "corpus"):
            a.corpus = _corpora(a.corpus)
        {"write": cmd_write, "verify": cmd_verify, "cpt-args": cmd_cpt_args, "sft-args": cmd_sft_args,
         "check-cpt": cmd_check_cpt, "check-pair": cmd_check_pair, "check-eval": cmd_check_eval,
         "adapter-digest": cmd_adapter_digest}[a.cmd](a)
    except LockError as e:
        print(f"REFUSED ({a.cmd}): {len(e.problems)} problem(s)", file=sys.stderr)
        for p in e.problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
