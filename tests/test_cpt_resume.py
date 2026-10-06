"""Falsifiable tests for Stage 4.3 mid-run resume (laptop, CPU, single process).

Fixture: a 1-layer ESM-2 MLM built locally (no network) with the ESM vocab, and a synthetic
``<eos>``-joined dataset whose row lengths span four buckets so the token budget yields a mixed
micro-batch plan. Masking noise is NOT replayed on resume (by design), so these tests check what
resume makes exact — data order, adapter / optimizer / scheduler state, counters — plus that a
resumed run finishes the same step budget. Each test says what a failure would look like.

Run: PYTHONPATH=. python -m pytest tests/test_cpt_resume.py -q   (needs transformers, peft, datasets)
"""
import json
import os
import sys

import numpy as np
import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "training"))
import cpt_training_lib as L  # noqa: E402
from quest.training.samplers import TokenBudgetStepSampler  # noqa: E402

VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
         "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]
AA = list("LAGVSERTIDPKQNFYMHWC")
ROWS_PER_STEP, STEPS_PER_EPOCH, SAVE_EVERY = 16, 15, 5


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("s43")
    model_dir = root / "esm_tiny"
    model_dir.mkdir()
    (model_dir / "vocab.txt").write_text("\n".join(VOCAB) + "\n")
    tok = EsmTokenizer(vocab_file=str(model_dir / "vocab.txt"))
    tok.save_pretrained(str(model_dir))
    torch.manual_seed(0)
    EsmForMaskedLM(EsmConfig(
        vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=256, pad_token_id=tok.pad_token_id,
        mask_token_id=tok.mask_token_id, position_embedding_type="rotary",
    )).save_pretrained(str(model_dir))

    rng = np.random.default_rng(0)

    def rows(n):  # 1-3 segments of 8-40 AAs -> 10..125 tokens, spanning buckets <32,<64,<128,>=128
        return ["<eos>".join("".join(rng.choice(AA, size=rng.integers(8, 41))) for _ in range(rng.integers(1, 4)))
                for _ in range(n)]

    ds = root / "ds"
    # 247 rows: 15 steps of 16 rows, 7-row remainder dropped from the short end.
    Dataset.from_dict({"input_text": rows(ROWS_PER_STEP * STEPS_PER_EPOCH + 7)}).save_to_disk(str(ds / "train"))
    Dataset.from_dict({"input_text": rows(40)}).save_to_disk(str(ds / "validation"))
    canon = root / "canon"
    Dataset.from_dict({"input_text": rows(32), "stratum_key": ["a", "b"] * 16}).save_to_disk(str(canon))
    return {"root": root, "model": str(model_dir), "ds": str(ds), "canon": str(canon)}


def make_cfg(world, out, **kw):
    base = dict(
        base_model=world["model"], dataset_path=world["ds"], canonical_eval_path=world["canon"],
        output_dir=str(out), lora_r=4, lora_alpha=8, learning_rate=1e-3, warmup_ratio=0.2, num_epochs=1,
        batch_size=4, gradient_accumulation_steps=4, eval_every_n_steps=SAVE_EVERY, save_every_n_steps=SAVE_EVERY,
        logging_steps=1, max_length=160, eos_strategy="fixb", attn_implementation="sdpa", use_compile=False,
        token_budget=512, bucket_boundaries=[32, 64, 128], dataloader_num_workers=0, eval_batch_size=16,
        region_eval=False, report_to="none", seed=7,
    )
    base.update(kw)
    return L.CPTConfig(**base)


@pytest.fixture
def seen(monkeypatch):
    """Records every training micro-batch as a list of input_ids tuples, in loader order."""
    log, orig = [], L.DataCollatorForMLMDynamic.__call__

    def rec(self, examples):
        log.append([tuple(ex["input_ids"]) for ex in examples])
        return orig(self, examples)

    monkeypatch.setattr(L.DataCollatorForMLMDynamic, "__call__", rec)
    return log


def _run(world, out, seen, **kw):
    seen.clear()
    tr = L.CPTTrainer(make_cfg(world, out, **kw))
    tr.train()
    return tr, list(seen)


@pytest.fixture(scope="module")
def run_a(world):
    """One uninterrupted token-budget run; its step-5 checkpoint is what the resume tests use."""
    log, orig = [], L.DataCollatorForMLMDynamic.__call__

    def rec(self, examples):
        log.append([tuple(ex["input_ids"]) for ex in examples])
        return orig(self, examples)

    L.DataCollatorForMLMDynamic.__call__ = rec
    try:
        out = world["root"] / "runA"
        tr = L.CPTTrainer(make_cfg(world, out))
        plan = tr.train_sampler.step_plan()
        tr.train()
    finally:
        L.DataCollatorForMLMDynamic.__call__ = orig
    return {"out": out, "trainer": tr, "plan": plan, "seen": list(log)}


def test_skip_sampler_yields_exactly_the_tail():
    """Failure: the wrapper drops the wrong batches, or len() disagrees with what it yields."""
    base = TokenBudgetStepSampler(np.full(64, 20), rows_per_step=16, token_budget=160, num_replicas=1, rank=0)
    full = list(iter(base))
    w = L._SkipSampler(base)
    assert list(iter(w)) == full and len(w) == len(full)
    w.skip = 3
    assert list(iter(w)) == full[3:] and len(w) == len(full) - 3


def test_run_a_is_a_real_token_budget_run(run_a):
    """Guards the fixture: more than one micro-batch per step (so skipping by micro-batch count is
    not the same as skipping by step), 15 steps, 3 resumable saves. Under the v2 sampler every
    step of this small fixture cuts the same way, so the plan is uniform; that is fine."""
    plan = run_a["plan"]
    assert len(plan) == STEPS_PER_EPOCH and max(plan) > 1, plan
    assert sum(plan) == len(run_a["seen"])
    for s in (5, 10, 15):
        assert (run_a["out"] / f"checkpoint_step_{s}" / L.RESUME_STATE_FILE).exists()
    assert not (run_a["out"] / "checkpoint_best" / L.RESUME_STATE_FILE).exists()
    st = torch.load(run_a["out"] / "checkpoint_step_5" / L.RESUME_STATE_FILE, weights_only=True)
    assert (st["global_step"], st["epoch"], st["step_in_epoch"]) == (5, 0, 5)


def test_resume_restores_state_exactly_then_continues_the_same_data_order(world, run_a, seen):
    """Failure: any restored quantity differs from the checkpoint (adapter bitwise, optimizer
    moments, LR position, best metric), or the resumed run's batches are not exactly the
    uninterrupted run's batches from step 6 on (off-by-one skip, wrong epoch seed)."""
    ck = run_a["out"] / "checkpoint_step_5"
    st = torch.load(ck / L.RESUME_STATE_FILE, weights_only=True)
    out_b = world["root"] / "runB"
    seen.clear()
    tr_b = L.CPTTrainer(make_cfg(world, out_b, resume_from=str(ck)))

    assert tr_b.global_step == 5 and tr_b.scheduler.last_epoch == 5
    lam = run_a["trainer"].scheduler.lr_lambdas[0]
    lr0 = make_cfg(world, out_b).learning_rate
    assert tr_b.scheduler.get_last_lr()[0] == pytest.approx(lam(5) * lr0)
    assert lam(5) != lam(0), "fixture must put step 5 at a different LR than a fresh start"
    assert tr_b.best_eval_loss == st["best_eval_loss"] < float("inf")

    sd = load_file(str(ck / "adapter_model.safetensors"))
    params = {n.replace(".default.", "."): p for n, p in tr_b.model.named_parameters() if p.requires_grad}
    assert set(params) == set(sd)
    assert all(torch.equal(params[n].detach().cpu(), sd[n]) for n in sd), "adapter not bitwise restored"
    live = tr_b.optimizer.state_dict()["state"]
    assert set(live) == set(st["optimizer"]["state"]) and len(live) == len(sd)
    assert all(torch.equal(live[i]["exp_avg"], st["optimizer"]["state"][i]["exp_avg"]) for i in live)

    tr_b.train()
    plan = run_a["plan"]
    assert list(seen) == run_a["seen"][sum(plan[:5]):], "resumed batches != uninterrupted batches after step 5"
    fin = json.load(open(out_b / "final" / "trainer_state.json"))
    assert fin["global_step"] == STEPS_PER_EPOCH
    assert (out_b / "checkpoint_step_10").exists() and (out_b / "checkpoint_step_15").exists()


def test_resume_rejects_config_drift_but_allows_cadence_and_gpu_count_changes(world, run_a):
    """Failure: a changed optimisation/data field resumes silently; or a harmless change
    (eval cadence, or the same rows/step from a different batch_size x accumulation — i.e. a
    different GPU count) is refused."""
    ck = str(run_a["out"] / "checkpoint_step_5")
    with pytest.raises(ValueError, match="learning_rate"):
        L.CPTTrainer(make_cfg(world, world["root"] / "drift1", resume_from=ck, learning_rate=2e-3))
    with pytest.raises(ValueError, match="rows_per_step"):
        L.CPTTrainer(make_cfg(world, world["root"] / "drift2", resume_from=ck, batch_size=8))
    ok = L.CPTTrainer(make_cfg(world, world["root"] / "ok", resume_from=ck, batch_size=8,
                               gradient_accumulation_steps=2, eval_every_n_steps=3))
    assert ok.global_step == 5


def test_resolve_resume_dir(tmp_path):
    """Failure: 'auto' picks a checkpoint without training_state.pt, orders by name (step_9 >
    step_10), trips over a stray non-numeric dir, starts FRESH over an old-format run (whose
    higher-numbered dirs would then make the cleanup delete the new run's own checkpoints), or an
    explicit old-format checkpoint resumes silently instead of raising."""
    assert L.resolve_resume_dir(None, str(tmp_path)) is None
    assert L.resolve_resume_dir("auto", str(tmp_path)) is None
    (tmp_path / "checkpoint_step_100").mkdir()          # old format: no state file
    with pytest.raises(FileNotFoundError, match="none of the 1 step checkpoints"):
        L.resolve_resume_dir("auto", str(tmp_path))
    for s in (9, 10):
        d = tmp_path / f"checkpoint_step_{s}"
        d.mkdir()
        (d / L.RESUME_STATE_FILE).write_bytes(b"x")
    (tmp_path / "checkpoint_step_10.bak").mkdir()
    (tmp_path / "checkpoint_step_10.bak" / L.RESUME_STATE_FILE).write_bytes(b"x")
    assert L.resolve_resume_dir("auto", str(tmp_path)) == str(tmp_path / "checkpoint_step_10")
    assert [s for s, _ in L._step_checkpoints(str(tmp_path))] == [9, 10, 100]
    with pytest.raises(FileNotFoundError, match="not a resumable checkpoint"):
        L.resolve_resume_dir(str(tmp_path / "checkpoint_step_100"), str(tmp_path))


def test_resume_on_legacy_bucketed_path_continues_data_order(world, seen):
    """The Stage 4.1 sampler path (token_budget=0). Failure: batches after resume differ from the
    uninterrupted run's — the index skip is miscounted (steps x accumulation x batch_size)."""
    kw = dict(token_budget=0, use_length_bucketing=True, seed=11)
    tr_a, seen_a = _run(world, world["root"] / "legA", seen, **kw)
    ga = tr_a.cfg.gradient_accumulation_steps
    ck = world["root"] / "legA" / "checkpoint_step_5"
    assert (ck / L.RESUME_STATE_FILE).exists()
    tr_b, seen_b = _run(world, world["root"] / "legB", seen, resume_from=str(ck), **kw)
    assert tr_b.global_step == STEPS_PER_EPOCH and seen_b == seen_a[5 * ga:]


def test_resume_refuses_the_random_sampler_path(world, seen):
    """Failure: a run with no deterministic sampler 'resumes' by replaying random data, or the
    refusal comes only after train() starts instead of at construction."""
    kw = dict(token_budget=0, use_length_bucketing=False, seed=5)
    _run(world, world["root"] / "rndA", seen, **kw)
    ck = world["root"] / "rndA" / "checkpoint_step_5"
    with pytest.raises(RuntimeError, match="deterministic sampler"):
        L.CPTTrainer(make_cfg(world, world["root"] / "rndB", resume_from=str(ck), **kw))


def test_resave_over_older_checkpoint_never_leaves_mixed_state(world, run_a, seen):
    """A resumed run re-saves checkpoint_step_10 over run A's. Failure: the old training_state.pt
    survives the adapter overwrite (a kill between the two would leave a new adapter beside the
    old state), i.e. the state file is not refreshed."""
    import shutil
    out = world["root"] / "runA_copy"
    shutil.copytree(run_a["out"], out)
    old_state = (out / "checkpoint_step_10" / L.RESUME_STATE_FILE).read_bytes()
    seen.clear()
    tr = L.CPTTrainer(make_cfg(world, out, resume_from=str(out / "checkpoint_step_5")))  # same output_dir
    tr.train()
    new_state = (out / "checkpoint_step_10" / L.RESUME_STATE_FILE).read_bytes()
    assert new_state != old_state
    assert torch.load(out / "checkpoint_step_10" / L.RESUME_STATE_FILE, weights_only=True)["global_step"] == 10
