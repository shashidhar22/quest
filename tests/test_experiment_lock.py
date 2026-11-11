"""The pilot's experiment lock: built from artifacts, verified against them, and the only source
of the trainer's and the judge's arguments.

Plan step 4 (plans/step12-diagnostic-and-pilot.md) and new-cpt-plan's `experiment-lock.json`
paragraph. Every refusal here is one way a pilot could run off the lock and still look locked.
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "training" / "pilot12_box"))
import experiment_lock as L  # noqa: E402
from track_a.shared.artifacts import dir_digest  # noqa: E402  (path set by the import above)

TRAINER = REPO / "scripts" / "training" / "train_cpt_stage2_single.py"
JUDGE = REPO / "track_a" / "esm_sft" / "judge.py"
HEX = "ab" * 32


# ------------------------------------------------------------------------------ fixtures
@pytest.fixture
def world(tmp_path):
    """A repo copy holding the locked files, two published markers, and a splits directory."""
    repo = tmp_path / "repo"
    for rel in L.code_files(REPO):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    corpora = {}
    for arm, version in L.ARM_CORPUS.items():
        d = tmp_path / f"corpus_{arm}"
        (d / "dataset").mkdir(parents=True)
        (d / "PUBLISHED.json").write_text(json.dumps(
            {"run_id": f"run-{arm}", "contract": {"sha256": "c" * 64, "version": version}}))
        corpora[arm] = d
    splits = tmp_path / "splits"
    splits.mkdir()
    for cond in L.BENCHMARK_CONDITIONS:
        (splits / f"as_paired_i_{cond}.parquet").write_text(f"stand-in {cond}\n")
    (splits / "as_class_i_candidate_pool.tsv").write_text("peptide\nAAAAAAAAA\n")
    lock = tmp_path / "experiment-lock.json"
    return {"repo": repo, "corpora": corpora, "splits": splits, "lock": lock, "tmp": tmp_path}


def common(w):
    return ["--corpus", f"P={w['corpora']['P']}", "--corpus", f"E={w['corpora']['E']}",
            "--splits-dir", str(w["splits"]), "--repo", str(w["repo"])]


def write(w, *extra):
    return L.main(["write", "--out", str(w["lock"]), *common(w), *extra])


def verify(w):
    return L.main(["verify", "--lock", str(w["lock"]), *common(w)])


@pytest.fixture
def locked(world):
    assert write(world) == 0
    world["doc"] = json.loads(world["lock"].read_text())
    return world


# ------------------------------------------------------------------------------ write / verify
def test_write_then_verify_round_trips_and_prints_the_lock_digest(locked, capsys):
    capsys.readouterr()
    assert verify(locked) == 0
    out = capsys.readouterr().out
    assert f"lock sha256: {L.sha256_file(locked['lock'])}" in out
    doc = locked["doc"]
    assert doc["schema"] == "experiment-lock/v1" and doc["stage"] == "pilot-10pct"
    assert sorted(doc["code_files"]) == sorted(L.code_files(REPO))
    assert doc["corpora"]["P"]["version"] == "repaired_proportional_10M"
    assert doc["corpora"]["E"]["run_id"] == "run-E"
    assert doc["corpora"]["P"]["marker_sha256"] == L.sha256_file(locked["corpora"]["P"] / "PUBLISHED.json")
    assert set(doc["benchmark_split_sha256"]) == {f"as_paired_i_{c}.parquet" for c in L.BENCHMARK_CONDITIONS}
    assert doc["cpt"]["gradient_accumulation_steps"] == {"P": 64, "E": 47}
    assert doc["sft"]["recipe"]["lr"] == 1e-4
    assert set(doc["evaluation"]["split_digests"]) == {"train", "val"}
    assert len(doc["evaluation"]["pool_digest"]) == 16          # the judge's own digest form


def test_write_refuses_to_overwrite(locked):
    before = locked["lock"].read_text()
    assert write(locked) != 0
    assert locked["lock"].read_text() == before


def test_write_refuses_a_corpus_given_for_the_wrong_arm(world):
    p, e = world["corpora"]["P"], world["corpora"]["E"]
    rc = L.main(["write", "--out", str(world["lock"]), "--corpus", f"P={e}", "--corpus", f"E={p}",
                 "--splits-dir", str(world["splits"]), "--repo", str(world["repo"])])
    assert rc != 0 and not world["lock"].exists()


def test_write_refuses_benchmark_hashes_that_are_not_the_splits_on_disk(world, tmp_path):
    j = tmp_path / "bench.json"
    j.write_text(json.dumps({"as_paired_i_train.parquet": "0" * 64}))
    assert write(world, "--benchmark-sha256", str(j)) != 0
    assert not world["lock"].exists()


def test_write_refuses_a_marker_missing_an_identity_field(world):
    (world["corpora"]["E"] / "PUBLISHED.json").write_text(json.dumps(
        {"contract": {"sha256": "c" * 64, "version": "task_enriched_cap1000"}}))
    assert write(world) != 0 and not world["lock"].exists()


@pytest.mark.parametrize("change, expect", [
    ("code", "code_files.track_a/esm_sft/judge.py"),
    ("marker", "corpora.E.marker_sha256"),
    ("split", "evaluation.split_digests.val"),
    ("recipe", "sft.recipe"),
])
def test_verify_refuses_each_change(locked, capsys, change, expect):
    if change == "code":
        p = locked["repo"] / "track_a/esm_sft/judge.py"
        p.write_text(p.read_text() + "\n# edited\n")
    elif change == "marker":
        m = locked["corpora"]["E"] / "PUBLISHED.json"
        m.write_text(m.read_text() + " ")
    elif change == "split":
        (locked["splits"] / "as_paired_i_val.parquet").write_text("other bytes\n")
    else:
        s = locked["repo"] / "track_a/esm_sft/scoring.py"
        s.write_text(s.read_text().replace('"lr": 1e-4', '"lr": 2e-4'))
    capsys.readouterr()
    assert verify(locked) != 0
    err = capsys.readouterr().err
    assert expect in err, err


def test_verify_reports_every_mismatch_at_once(locked, capsys):
    (locked["repo"] / "scripts/training/pilot12_box/pilot_compare.py").unlink()
    run_id_changed = json.loads((locked["corpora"]["P"] / "PUBLISHED.json").read_text())
    run_id_changed["run_id"] = "another-build"
    (locked["corpora"]["P"] / "PUBLISHED.json").write_text(json.dumps(run_id_changed))
    (locked["splits"] / "as_class_i_candidate_pool.tsv").write_text("peptide\nCCCCCCCCC\n")
    capsys.readouterr()
    assert verify(locked) != 0
    err = capsys.readouterr().err
    for needle in ("code_files.scripts/training/pilot12_box/pilot_compare.py",
                   "corpora.P.run_id", "corpora.P.marker_sha256", "evaluation.pool_digest"):
        assert needle in err, err


def test_verify_refuses_a_hand_edited_frozen_value(locked, capsys):
    doc = locked["doc"]
    doc["cpt"]["lr"] = 3e-4
    locked["lock"].write_text(json.dumps(doc, indent=1))
    capsys.readouterr()
    assert verify(locked) != 0
    assert "cpt.lr" in capsys.readouterr().err


# ------------------------------------------------------------------------------ cpt-args
def cpt_args(w, arm, capsys, *extra):
    capsys.readouterr()
    rc = L.main(["cpt-args", "--lock", str(w["lock"]), "--arm", arm,
                 "--dataset-path", str(w["corpora"][arm] / "dataset"),
                 "--output-dir", str(w["tmp"] / f"cpt_{arm}"), "--canonical-eval-path", "/c",
                 *(extra or ("--benchmark-sha256", str(w["tmp"] / "bench_from_lock.json")))])
    assert rc == 0, capsys.readouterr().err
    return capsys.readouterr().out.splitlines()


def real_trainer_parser():
    """The trainer's own `parse_args`, executed from its source: importing the module would load
    torch and the CPT library, which is not a side effect a parser test should depend on."""
    tree = ast.parse(TRAINER.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
    ns = {"argparse": argparse}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(TRAINER), "exec"), ns)
    return ns["parse_args"]


def parse_with_trainer(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_cpt_stage2_single.py", *argv])
    return real_trainer_parser()()


def test_cpt_args_differ_between_arms_only_in_accumulation_and_paths(locked, capsys, monkeypatch):
    p = cpt_args(locked, "P", capsys)
    e = cpt_args(locked, "E", capsys)
    assert len(p) == len(e)
    diff = {(a, b) for a, b in zip(p, e) if a != b}
    assert diff == {("64", "47"),
                    (str(locked["corpora"]["P"] / "dataset"), str(locked["corpora"]["E"] / "dataset")),
                    (str(locked["tmp"] / "cpt_P"), str(locked["tmp"] / "cpt_E"))}
    ap = parse_with_trainer(p, monkeypatch)
    ae = parse_with_trainer(e, monkeypatch)
    assert (ap.gradient_accumulation_steps, ae.gradient_accumulation_steps) == (64, 47)
    for a in (ap, ae):
        assert a.max_steps == 780 and a.eval_steps == 781 and a.save_steps == 780   # no eval before the final one
        assert a.max_length == 1024 and a.bucket_boundaries == [64, 128, 256, 512, 1024]
        assert a.tcr_variant == "cdr3" and a.mhc_variant == "full" and a.token_budget == 8192
        assert a.loss_weighting == "token_exact" and a.corpus_preflight == "required"
        assert a.early_stop_n_patience == 0 and a.report_to == "none" and a.no_compile
        assert a.lr == 2e-4 and a.warmup_ratio == 0.05 and a.weight_decay == 0.01
        assert a.gradient_clip == 1.0 and a.mlm_probability == 0.15 and a.batch_size == 16
        assert a.lora_r == 32 and a.lora_alpha == 64 and a.lora_dropout == 0.05
        assert a.seed == 42 and a.eos_strategy == "raw" and a.base_model == "esmc_300m"
        assert a.num_epochs == 1 and a.canonical_eval_path == "/c"
        # Always on, and written from the lock itself when the file is not there yet.
        assert json.loads(Path(a.corpus_benchmark_sha256).read_text()) == \
            locked["doc"]["benchmark_split_sha256"]


def test_every_legacy_defaulted_flag_is_set_explicitly(locked, capsys):
    argv = cpt_args(locked, "P", capsys)
    for flag in ("--max_length", "--bucket_boundaries", "--tcr_variant", "--mhc_variant",
                 "--token_budget", "--loss_weighting", "--corpus_preflight", "--max_steps",
                 "--early_stop_n_patience", "--report_to", "--no_compile", "--eval_steps",
                 "--save_steps", "--lr", "--warmup_ratio", "--weight_decay", "--gradient_clip",
                 "--mlm_probability", "--lora_r", "--lora_alpha", "--lora_dropout", "--seed",
                 "--batch_size", "--gradient_accumulation_steps", "--eos_strategy",
                 "--base_model", "--num_epochs"):
        assert flag in argv, flag


def test_cpt_args_refuses_the_other_arms_corpus(locked, capsys):
    rc = L.main(["cpt-args", "--lock", str(locked["lock"]), "--arm", "P",
                 "--dataset-path", str(locked["corpora"]["E"] / "dataset"),
                 "--output-dir", str(locked["tmp"] / "cpt_P"), "--canonical-eval-path", "/c",
                 "--benchmark-sha256", str(locked["tmp"] / "b.json")])
    assert rc != 0 and not (locked["tmp"] / "b.json").exists()


def test_cpt_args_passes_benchmark_hashes_only_when_they_are_the_locked_ones(locked, capsys, monkeypatch):
    good = locked["tmp"] / "bench.json"
    good.write_text(json.dumps(locked["doc"]["benchmark_split_sha256"]))
    argv = cpt_args(locked, "P", capsys, "--benchmark-sha256", str(good))
    assert parse_with_trainer(argv, monkeypatch).corpus_benchmark_sha256 == str(good)
    bad = locked["tmp"] / "bad.json"
    bad.write_text(json.dumps({"as_paired_i_train.parquet": "0" * 64}))
    rc = L.main(["cpt-args", "--lock", str(locked["lock"]), "--arm", "P",
                 "--dataset-path", str(locked["corpora"]["P"] / "dataset"),
                 "--output-dir", str(locked["tmp"] / "cpt_P"), "--canonical-eval-path", "/c",
                 "--benchmark-sha256", str(bad)])
    assert rc != 0


# ------------------------------------------------------------------------------ sft-args
def test_sft_args_carry_the_locked_recipe_and_the_validation_protocol(locked, capsys):
    capsys.readouterr()
    assert L.main(["sft-args", "--lock", str(locked["lock"]), "--stage", "train",
                   "--adapter", "none", "--out-dir", "/o/sft_B"]) == 0
    train = capsys.readouterr().out.splitlines()
    assert train[0] == "train"
    pairs = dict(zip(train[1::2], train[2::2]))
    assert pairs["--adapter"] == "none" and pairs["--out_dir"] == "/o/sft_B"
    assert pairs["--task"] == "as_paired_i" and pairs["--mhc_variant"] == "full"
    recipe = locked["doc"]["sft"]["recipe"]
    for k, v in recipe.items():
        assert pairs[f"--{k}"] == str(v)
    assert L.main(["sft-args", "--lock", str(locked["lock"]), "--stage", "eval", "--adapter", "/a/final",
                   "--ckpt", "/o/best.pt", "--out", "/o/eval.json"]) == 0
    ev = capsys.readouterr().out.splitlines()
    assert ev[0] == "eval"
    pairs = dict(zip(ev[1::2], ev[2::2]))
    assert pairs["--conditions"] == "val" and pairs["--distractor_draw"] == "length_matched"
    assert pairs["--max_queries"] == "5000" and pairs["--min_peptides"] == "100"
    assert pairs["--ckpt"] == "/o/best.pt" and pairs["--out"] == "/o/eval.json"
    assert pairs["--max_len"] == str(recipe["max_len"])
    # The judge's parser imports torch, so its flag names are checked against its source.
    src = JUDGE.read_text()
    for flag in set(train[1::2]) | set(ev[1::2]):
        assert f'"{flag}"' in src, flag


# ------------------------------------------------------------------------------ check-cpt
def good_state(doc, arm):
    """A final/trainer_state.json as the trainer writes it, consistent with the lock."""
    c, m = doc["cpt"], doc["model"]
    cfg = {"base_model": m["base_model"], "lora_r": m["lora_r"], "lora_alpha": m["lora_alpha"],
           "lora_dropout": m["lora_dropout"], "learning_rate": c["lr"],
           "warmup_ratio": c["warmup_ratio"], "weight_decay": c["weight_decay"],
           "gradient_clip": c["gradient_clip"], "num_epochs": c["num_epochs"],
           "batch_size": c["batch_size"],
           "gradient_accumulation_steps": c["gradient_accumulation_steps"][arm],
           "eval_every_n_steps": c["max_steps"] + 1, "save_every_n_steps": c["max_steps"],
           "max_length": c["max_length"], "mlm_probability": c["mlm_probability"],
           "eos_strategy": c["eos_strategy"], "use_compile": False,
           "bucket_boundaries": c["bucket_boundaries"], "token_budget": c["token_budget"],
           "loss_weighting": c["loss_weighting"], "max_steps": c["max_steps"],
           "corpus_preflight": "required", "tcr_variant": c["tcr_variant"],
           "mhc_variant": c["mhc_variant"], "early_stop_n_patience": c["early_stop_n_patience"],
           "report_to": "none", "seed": c["seed"], "logging_steps": 50}
    ident = doc["corpora"][arm]
    return {"global_step": c["max_steps"], "masked_targets_seen": c["pilot_target_budget"],
            "initial_lora_sha256": HEX,
            "corpus_identity": {"run_id": ident["run_id"], "contract_sha256": ident["contract_sha256"],
                                "contract_version": ident["version"],
                                "marker_sha256": ident["marker_sha256"], "benchmark_checked": False},
            "loss_weighting_semantics": c["loss_weighting_semantics"], "config": cfg}


def put_state(tmp, name, state, weights=True):
    d = tmp / name / "final"
    d.mkdir(parents=True, exist_ok=True)
    (d / "trainer_state.json").write_text(json.dumps(state))
    if weights:
        (d / "adapter_model.safetensors").write_bytes(safetensors_bytes())
        (d / "adapter_config.json").write_text('{"r": 32}')
    return d


def check_cpt(w, arm, final):
    return L.main(["check-cpt", "--lock", str(w["lock"]), "--arm", arm, "--final", str(final)])


def test_check_cpt_passes_a_run_that_matches_the_lock(locked):
    for arm in ("P", "E"):
        assert check_cpt(locked, arm, put_state(locked["tmp"], f"cpt_{arm}", good_state(locked["doc"], arm))) == 0


def _drop_init(s):
    s.pop("initial_lora_sha256")


def _drop_cfg(s):
    s["config"].pop("token_budget")


@pytest.mark.parametrize("break_it, expect", [
    (lambda s: s.update(global_step=779), "global_step"),
    (lambda s: s.update(loss_weighting_semantics="row-share/v1"), "loss_weighting_semantics"),
    (lambda s: s["corpus_identity"].update(run_id="run-E"), "corpus_identity.run_id"),
    (_drop_init, "initial_lora_sha256"),
    (lambda s: s.update(initial_lora_sha256=None), "initial_lora_sha256"),
    (lambda s: s["config"].update(learning_rate=1e-4), "config.learning_rate"),
    (lambda s: s["config"].update(gradient_accumulation_steps=47), "config.gradient_accumulation_steps"),
    (_drop_cfg, "config.token_budget"),
    (lambda s: s.update(masked_targets_seen=0), "masked_targets_seen"),
])
def test_check_cpt_refuses(locked, capsys, break_it, expect):
    state = good_state(locked["doc"], "P")
    break_it(state)
    capsys.readouterr()
    assert check_cpt(locked, "P", put_state(locked["tmp"], "cpt_P", state)) != 0
    assert expect in capsys.readouterr().err


def test_check_cpt_refuses_a_missing_state_file(locked):
    assert check_cpt(locked, "P", locked["tmp"] / "nowhere") != 0


# ------------------------------------------------------------------------------ check-pair
def pair(w, capsys, p_over=None, e_over=None):
    p, e = good_state(w["doc"], "P"), good_state(w["doc"], "E")
    p.update(p_over or {})
    e.update(e_over or {})
    capsys.readouterr()
    rc = L.main(["check-pair", "--lock", str(w["lock"]),
                 "--final-p", str(put_state(w["tmp"], "cpt_P", p)),
                 "--final-e", str(put_state(w["tmp"], "cpt_E", e))])
    out = capsys.readouterr()
    return rc, out.out + out.err


def test_check_pair_passes_a_good_pair_and_prints_the_counts(locked, capsys):
    rc, text = pair(locked, capsys, {"masked_targets_seen": 2_300_000}, {"masked_targets_seen": 2_320_000})
    assert rc == 0, text
    assert "2300000" in text and "2320000" in text and "relative gap" in text


@pytest.mark.parametrize("p_over, e_over", [
    ({}, {"initial_lora_sha256": "cd" * 32}),
    ({"masked_targets_seen": 2_300_000}, {"masked_targets_seen": 2_250_000}),   # 2.2% apart
    ({"masked_targets_seen": 2_000_000}, {"masked_targets_seen": 2_000_000}),   # equal, off budget
    ({"initial_lora_sha256": None}, {"initial_lora_sha256": None}),             # absent on both
])
def test_check_pair_refuses(locked, capsys, p_over, e_over):
    rc, text = pair(locked, capsys, p_over, e_over)
    assert rc != 0, text


# ------------------------------------------------------------------------------ check-eval
def good_eval(doc, arm, tmp):
    ev, sft = doc["evaluation"], doc["sft"]
    recipe = sft["recipe"]
    keys = ("lr", "batch_size", "epochs", "patience", "max_train_pos", "max_val_pos", "num_workers")
    if arm == "B":
        lineage = {"cpt_adapter": "none", "cpt_objective": "none", "cpt_steps": None,
                   "cpt_seed": None, "cpt_token_budget": None}
    else:
        final = tmp / f"cpt_{arm}" / "final"
        final.mkdir(parents=True, exist_ok=True)
        (final / "adapter_model.safetensors").write_bytes(safetensors_bytes((1.0, float(ord(arm)))))
        lineage = {"cpt_adapter": str(final), "cpt_adapter_digest": dir_digest(final),
                   "cpt_objective": doc["cpt"]["loss_weighting_semantics"],
                   "cpt_steps": doc["cpt"]["max_steps"], "cpt_seed": doc["cpt"]["seed"],
                   "cpt_token_budget": doc["cpt"]["token_budget"]}
    return {"args": {"task": sft["task"], "conditions": ["val"], "distractor_draw": ev["distractor_draw"],
                     "smoke": False,
                     "max_queries": ev["max_queries"], "min_peptides": ev["min_peptides"]},
            "pool_size": 100,
            "conditions": {"val": {"ranks": {"r1": 3}, "true_peptides": {"r1": "AAAAAAAAA"}}},
            "provenance": {"pool_digest": ev["pool_digest"],
                           "split_digests": {"val": ev["split_digests"]["val"]},
                           "sft_train_digests": dict(ev["split_digests"]),
                           "sft_seed": sft["seed"], "sft_max_len": recipe["max_len"],
                           "sft_recipe": ";".join(f"{k}={recipe[k]}" for k in keys),
                           **lineage}}


def check_eval(w, arm, doc, capsys):
    p = w["tmp"] / f"eval_{arm}.json"
    p.write_text(json.dumps(doc))
    capsys.readouterr()
    final = () if arm == "B" else ("--final", str(w["tmp"] / f"cpt_{arm}" / "final"))
    rc = L.main(["check-eval", "--lock", str(w["lock"]), "--eval", str(p), "--arm", arm, *final])
    return rc, capsys.readouterr().err


def test_check_eval_passes_each_good_cell(locked, capsys):
    for arm in ("B", "P", "E"):
        rc, err = check_eval(locked, arm, good_eval(locked["doc"], arm, locked["tmp"]), capsys)
        assert rc == 0, err


def _scored_test(d):
    d["conditions"]["test_novel_pep"] = {"ranks": {"r1": 1}}


def _wrong_draw(d):
    d["args"]["distractor_draw"] = "uniform"


def _old_objective(d):
    d["provenance"]["cpt_objective"] = "fixed-batch-mean/v0"


def _short_cpt(d):
    d["provenance"]["cpt_steps"] = 7800


def _other_val(d):
    d["provenance"]["split_digests"] = {"val": "0" * 16}


def _other_train(d):
    d["provenance"]["sft_train_digests"]["train"] = "0" * 16


def _other_seed(d):
    d["provenance"]["sft_seed"] = 43


def _other_recipe(d):
    d["provenance"]["sft_recipe"] = d["provenance"]["sft_recipe"].replace("lr=0.0001", "lr=0.0002")


@pytest.mark.parametrize("arm, break_it, expect", [
    ("B", _scored_test, "conditions"),
    ("P", _wrong_draw, "distractor_draw"),
    ("E", _old_objective, "cpt_objective"),
    ("P", _short_cpt, "cpt_steps"),
    ("B", _other_val, "split_digests"),
    ("E", _other_train, "sft_train_digests"),
    ("B", _other_seed, "sft_seed"),
    ("P", _other_recipe, "sft_recipe"),
])
def test_check_eval_refuses(locked, capsys, arm, break_it, expect):
    doc = good_eval(locked["doc"], arm, locked["tmp"])
    break_it(doc)
    rc, err = check_eval(locked, arm, doc, capsys)
    assert rc != 0 and expect in err, err


def test_check_eval_refuses_b_with_an_adapter_and_p_without_one(locked, capsys):
    rc, err = check_eval(locked, "B", good_eval(locked["doc"], "P", locked["tmp"]), capsys)
    assert rc != 0 and "cpt_adapter" in err
    rc, err = check_eval(locked, "P", good_eval(locked["doc"], "B", locked["tmp"]), capsys)
    assert rc != 0 and "cpt_adapter" in err


def test_check_eval_binds_a_cell_to_its_own_arms_adapter(locked, capsys):
    """Both arms record the same objective, steps, seed and budget; only the adapter tells them apart."""
    good_eval(locked["doc"], "P", locked["tmp"])                       # creates cpt_P/final
    swapped = good_eval(locked["doc"], "E", locked["tmp"])             # an eval made with E's adapter
    rc, err = check_eval(locked, "P", swapped, capsys)                 # ...presented as cell P
    assert rc != 0 and "cpt_adapter_digest" in err
    p = locked["tmp"] / "eval_noflag.json"
    p.write_text(json.dumps(good_eval(locked["doc"], "P", locked["tmp"])))
    assert L.main(["check-eval", "--lock", str(locked["lock"]), "--eval", str(p), "--arm", "P"]) != 0


def test_absence_on_both_sides_is_a_mismatch():
    problems = []
    L._same("field", None, None, problems)
    assert problems and "field" in problems[0]


def test_check_cpt_refuses_a_final_directory_without_adapter_weights(locked, capsys):
    """The trainer logs a failed adapter save and still writes trainer_state.json."""
    final = put_state(locked["tmp"], "cpt_noweights", good_state(locked["doc"], "P"), weights=False)
    capsys.readouterr()
    assert check_cpt(locked, "P", final) != 0
    assert "adapter_model.safetensors" in capsys.readouterr().err


def test_check_pair_refuses_arms_each_inside_the_budget_tolerance_but_apart(locked, capsys):
    rc, text = pair(locked, capsys, {"masked_targets_seen": 2_254_000}, {"masked_targets_seen": 2_346_000})
    assert rc != 0 and "apart in masked targets" in text
    assert "off the budget" not in text                   # only the arm-to-arm bound refused it


def safetensors_bytes(values=(0.5, -0.25)):
    """A minimal valid safetensors file: one float32 tensor."""
    import struct
    data = struct.pack(f"<{len(values)}f", *values)
    header = json.dumps({"lora_A": {"dtype": "F32", "shape": [len(values)], "data_offsets": [0, len(data)]}}).encode()
    return struct.pack("<Q", len(header)) + header + data


# ------------------------------------------------------------------------------ after the Codex review
def test_the_lock_pins_every_repo_module_the_trainer_and_the_judge_import():
    """Masking, batching and negative sampling live outside the files first listed."""
    files = set(L.code_files(REPO))
    for rel in ("quest/training/samplers.py", "quest/training/collators.py", "track_a/esm_sft/model.py",
                "track_a/esm_sft_cubes/cpt_loader.py", "track_a/shared/load_splits.py",
                "track_a/shared/negative_sampling.py", "track_a/esm_sft/encoders.py",
                "scripts/training/stage42_box/readability.py", "scripts/training/cpt_training_lib.py",
                "scripts/training/pilot12_box/run_pilot.sh", "scripts/training/pilot12_box/experiment_lock.py"):
        assert rel in files, rel


def test_verify_refuses_a_change_in_an_imported_module_and_a_new_import(locked, capsys):
    p = locked["repo"] / "track_a/shared/negative_sampling.py"
    keep = p.read_text()
    p.write_text(keep + "\n# edited\n")
    capsys.readouterr()
    assert verify(locked) != 0 and "negative_sampling.py" in capsys.readouterr().err
    p.write_text(keep)
    assert verify(locked) == 0
    (locked["repo"] / "track_a/shared/extra_rule.py").write_text("X = 1\n")
    j = locked["repo"] / "track_a/esm_sft/judge.py"
    j.write_text(j.read_text() + "\nfrom track_a.shared import extra_rule  # noqa\n")
    capsys.readouterr()
    assert verify(locked) != 0
    assert "extra_rule.py" in capsys.readouterr().err       # a file the lock never pinned


@pytest.mark.parametrize("content", ["{}", "null", "[]", "0", '""', "false"])
def test_an_empty_or_non_object_record_is_not_agreement(locked, capsys, content):
    final = put_state(locked["tmp"], "cpt_empty", good_state(locked["doc"], "P"))
    (final / "trainer_state.json").write_text(content)
    assert check_cpt(locked, "P", final) != 0
    ev = locked["tmp"] / "eval_empty.json"
    ev.write_text(content)
    assert L.main(["check-eval", "--lock", str(locked["lock"]), "--eval", str(ev), "--arm", "B"]) != 0
    other = put_state(locked["tmp"], "cpt_ok", good_state(locked["doc"], "E"))
    assert L.main(["check-pair", "--lock", str(locked["lock"]), "--final-p", str(final),
                   "--final-e", str(other)]) != 0


def _truncate(final):
    w = final / "adapter_model.safetensors"
    w.write_bytes(w.read_bytes()[:-3])


def _nan(final):
    (final / "adapter_model.safetensors").write_bytes(safetensors_bytes((1.0, float("nan"))))


def _no_config(final):
    (final / "adapter_config.json").unlink()


@pytest.mark.parametrize("damage, expect", [(_truncate, "adapter_model.safetensors"), (_nan, "non-finite"),
                                            (_no_config, "adapter_config.json")])
def test_check_cpt_refuses_an_adapter_that_is_incomplete_or_not_finite(locked, capsys, damage, expect):
    final = put_state(locked["tmp"], "cpt_damaged", good_state(locked["doc"], "P"))
    damage(final)
    capsys.readouterr()
    assert check_cpt(locked, "P", final) != 0
    assert expect in capsys.readouterr().err


def test_a_smoke_evaluation_is_refused_and_no_argv_asks_for_one(locked, capsys):
    doc = good_eval(locked["doc"], "B", locked["tmp"])
    doc["args"]["smoke"] = True
    rc, err = check_eval(locked, "B", doc, capsys)
    assert rc != 0 and "smoke" in err
    doc["args"].pop("smoke")                               # not recorded is not "no"
    rc, err = check_eval(locked, "B", doc, capsys)
    assert rc != 0 and "smoke" in err
    for stage, extra in (("train", ["--out-dir", "/o"]), ("eval", ["--ckpt", "/c", "--out", "/e"])):
        capsys.readouterr()
        assert L.main(["sft-args", "--lock", str(locked["lock"]), "--stage", stage, "--adapter", "none", *extra]) == 0
        assert "--smoke" not in capsys.readouterr().out


def test_adapter_digest_names_the_directory_the_judge_would_record(locked, capsys):
    final = put_state(locked["tmp"], "cpt_d", good_state(locked["doc"], "P"))
    capsys.readouterr()
    assert L.main(["adapter-digest", "--final", str(final)]) == 0
    assert capsys.readouterr().out.strip() == dir_digest(final)
    assert L.main(["adapter-digest", "--final", str(locked["tmp"] / "nowhere")]) != 0


def test_the_lock_states_what_it_does_not_pin(locked):
    text = " ".join(locked["doc"]["limitations"])
    assert "model weights" in text and "canonical evaluation" in text
