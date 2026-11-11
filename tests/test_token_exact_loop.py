"""`token_exact` in the REAL training loop (corpus repair step 12, new-cpt-plan gate 5).

`tests/test_loss_normalization.py` proves the contract of the pieces. This file proves the loop
uses them: `exact_step_batches` against a concatenated reference with simulated ranks, then the
actual `CPTTrainer` on CPU, whose clipped gradient is compared with a reference recomputed from
the batches it consumed. The shipped `token` weighting is the control: it must NOT match, or the
comparison could pass vacuously.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import cpt_training_lib as L  # noqa: E402
from quest.data.corpus_preflight import MARKER, corpus_files  # noqa: E402
from quest.training.loss_normalization import (  # noqa: E402
    exact_step_batches,
    masked_target_count,
    summed_masked_loss,
)
from test_corpus_preflight import make as make_version_dir  # noqa: E402

VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
         "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]
AA = list("LAGVSERTIDPKQNFYMHWC")
ROWS_PER_STEP, STEPS, SAVE_EVERY = 16, 15, 5


# ------------------------------------------------------------------ the generator, tiny model
def tiny():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Embedding(12, 8), torch.nn.Linear(8, 12))


def micro_batch(rng, rows, length, n_targets):
    ids = torch.from_numpy(rng.integers(0, 12, size=(rows, length)))
    labels = torch.full((rows, length), -100)
    flat = rng.choice(rows * length, size=n_targets, replace=False)
    labels.view(-1)[torch.from_numpy(flat)] = ids.view(-1)[torch.from_numpy(flat)]
    return {"input_ids": ids, "labels": labels}


def grads(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


def reference_grad(batches):
    """One pass over everything: the summed loss of every target over the number of targets."""
    model = tiny()
    n = sum(masked_target_count(b["labels"]) for b in batches)
    (sum(summed_masked_loss(model(b["input_ids"]), b["labels"]) for b in batches) / n).backward()
    return grads(model)


@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_buffered_steps_give_the_concatenated_gradient_on_every_rank_layout(world_size):
    rng = np.random.default_rng(world_size)
    plan = [3, 1, 4]                                    # uneven micro-batch counts per step
    shapes = [(4, 9, 5), (1, 40, 0), (7, 3, 11), (2, 17, 6), (5, 5, 1), (3, 21, 20), (1, 6, 2), (6, 8, 9)]
    per_rank = [[micro_batch(rng, *shapes[(i + r) % len(shapes)]) for i in range(sum(plan))]
                for r in range(world_size)]
    assert any(masked_target_count(b["labels"]) == 0 for b in per_rank[0])   # the zero-target case is in

    def global_count(step_i):                           # what the all-reduce would return
        lo = sum(plan[:step_i])
        return lambda _local: sum(masked_target_count(b["labels"])
                                  for rank in per_rank for b in rank[lo:lo + plan[step_i]])

    for step_i, n_acc in enumerate(plan):
        lo = sum(plan[:step_i])
        rank_grads = []
        for rank in per_rank:
            model = tiny()
            seen = 0
            for batch, scale in exact_step_batches(rank[lo:lo + n_acc], [n_acc], world_size, global_count(step_i)):
                (summed_masked_loss(model(batch["input_ids"]), batch["labels"]) * scale).backward()
                seen += 1
            assert seen == n_acc
            rank_grads.append(grads(model))
        ddp = torch.stack(rank_grads).mean(0)           # DDP averages gradients over ranks
        want = reference_grad([b for rank in per_rank for b in rank[lo:lo + n_acc]])
        assert torch.isfinite(ddp).all()
        assert torch.allclose(ddp, want, rtol=1e-5, atol=1e-7), step_i


def test_the_loader_and_the_plan_must_agree():
    rng = np.random.default_rng(0)
    batches = [micro_batch(rng, 2, 5, 3) for _ in range(5)]
    assert len(list(exact_step_batches(batches, [2, 3]))) == 5
    with pytest.raises(RuntimeError, match="ended inside optimizer step 1"):
        list(exact_step_batches(batches, [2, 4]))
    with pytest.raises(RuntimeError, match="beyond the 2 planned steps"):
        list(exact_step_batches(batches, [2, 2]))


def test_a_step_with_no_target_anywhere_is_refused_not_divided():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="no supervised targets"):
        list(exact_step_batches([micro_batch(rng, 2, 5, 0), micro_batch(rng, 3, 4, 0)], [2]))


def test_every_micro_batch_of_a_step_gets_the_same_scale():
    rng = np.random.default_rng(0)
    batches = [micro_batch(rng, 2, 9, n) for n in (1, 7, 0, 4)]
    scales = [s for _b, s in exact_step_batches(batches, [3, 1], world_size=4)]
    assert scales == [4 / 8, 4 / 8, 4 / 8, 4 / 4]


# ------------------------------------------------------------------ the real trainer, on CPU
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("exact")
    model_dir = root / "esm_tiny"
    model_dir.mkdir()
    (model_dir / "vocab.txt").write_text("\n".join(VOCAB) + "\n")
    tok = EsmTokenizer(vocab_file=str(model_dir / "vocab.txt"))
    tok.save_pretrained(str(model_dir))
    torch.manual_seed(0)
    EsmForMaskedLM(EsmConfig(                           # no dropout: the reference must be exact
        vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=256, pad_token_id=tok.pad_token_id,
        mask_token_id=tok.mask_token_id, position_embedding_type="rotary",
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )).save_pretrained(str(model_dir))
    rng = np.random.default_rng(0)

    def rows(n):
        return ["<eos>".join("".join(rng.choice(AA, size=rng.integers(8, 41))) for _ in range(rng.integers(1, 4)))
                for _ in range(n)]

    version = make_version_dir(root / "corpus" / "v", publish=False)      # the audits and checks
    for split in ("train", "validation", "test"):
        for f in (version / "dataset" / split).iterdir():
            f.unlink()
    DatasetDict({"train": Dataset.from_dict({"input_text": rows(ROWS_PER_STEP * STEPS + 7)}),
                 "validation": Dataset.from_dict({"input_text": rows(40)}),
                 "test": Dataset.from_dict({"input_text": rows(8)})}).save_to_disk(str(version / "dataset"))
    (version / MARKER).write_text(json.dumps({
        "run_id": "r1", "contract": {"sha256": "c0", "version": "enriched"},
        "objects": {k: {"size": s, "sha256": h} for k, (s, h) in corpus_files(version).items()}}))
    canon = root / "canon"
    Dataset.from_dict({"input_text": rows(32), "stratum_key": ["a", "b"] * 16}).save_to_disk(str(canon))
    return {"root": root, "model": str(model_dir), "version": version, "ds": str(version / "dataset"),
            "canon": str(canon)}


def make_cfg(world, out, **kw):
    base = dict(
        base_model=world["model"], dataset_path=world["ds"], canonical_eval_path=world["canon"],
        output_dir=str(out), lora_r=4, lora_alpha=8, lora_dropout=0.0, learning_rate=1e-3, warmup_ratio=0.2,
        num_epochs=1, batch_size=4, gradient_accumulation_steps=4, eval_every_n_steps=SAVE_EVERY,
        save_every_n_steps=SAVE_EVERY, logging_steps=1, max_length=160, eos_strategy="fixb",
        attn_implementation="sdpa", use_compile=False, token_budget=512, bucket_boundaries=[32, 64, 128],
        dataloader_num_workers=0, eval_batch_size=16, region_eval=False, report_to="none", seed=7,
        loss_weighting="token_exact", corpus_preflight="required",
    )
    base.update(kw)
    return L.CPTConfig(**base)


def first_step(world, out, monkeypatch, **kw):
    """Run the trainer; return (first step's clipped gradient, the model as it was then, the
    micro-batches of that step, the trainer)."""
    batches, orig_collate = [], L.DataCollatorForMLMDynamic.__call__

    def collate(self, examples):
        batch = orig_collate(self, examples)
        batches.append({k: v.clone() for k, v in batch.items()})
        return batch

    captured, orig_clip = {}, torch.nn.utils.clip_grad_norm_

    def clip(params, max_norm, *a, **k):
        params = list(params)
        if not captured:
            captured["grad"] = torch.cat([p.grad.reshape(-1) for p in params if p.grad is not None]).clone()
            captured["model"] = copy.deepcopy(tr.model)
        return orig_clip(params, max_norm, *a, **k)

    monkeypatch.setattr(L.DataCollatorForMLMDynamic, "__call__", collate)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    tr = L.CPTTrainer(make_cfg(world, out, **kw))
    plan = tr.train_sampler.step_plan()
    tr.train()
    return captured["grad"], captured["model"], batches[:plan[0]], tr, plan


def exact_reference(model, batches):
    model.zero_grad()
    n = sum(masked_target_count(b["labels"]) for b in batches)
    total = sum(summed_masked_loss(
        model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float(), b["labels"])
        for b in batches)
    (total / n).backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])


def test_the_real_loop_clips_the_exact_global_mean_gradient(world, monkeypatch):
    got, model, batches, tr, plan = first_step(world, world["root"] / "exact", monkeypatch)
    counts = [masked_target_count(b["labels"]) for b in batches]
    assert len(batches) > 1 and len(set(counts)) > 1, "the step must be cut unevenly to mean anything"
    want = exact_reference(model, batches)
    assert got.shape == want.shape and torch.isfinite(got).all()
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-7)
    assert tr.global_step == len(plan) == STEPS                 # buffering moved no step boundary
    state = torch.load(world["root"] / "exact" / f"checkpoint_step_{SAVE_EVERY}" / L.RESUME_STATE_FILE,
                       weights_only=True)
    assert state["config"]["loss_weighting_semantics"] == "global-masked-target-mean/v2"
    assert state["config"]["corpus_identity"]["run_id"] == "r1"
    # Training wrote nothing into the published corpus: it still passes its own marker.
    assert L.check_corpus_and_objective(make_cfg(world, world["root"] / "again"))["run_id"] == "r1"
    assert list((world["root"] / "exact" / "tokenized_cache").glob("train-*.arrow"))


def test_the_shipped_token_weighting_is_not_that_gradient(world, monkeypatch):
    """The control: same data, the historical objective, and the comparison must fail."""
    got, model, batches, _tr, _plan = first_step(world, world["root"] / "token", monkeypatch,
                                                 loss_weighting="token")
    assert not torch.allclose(got, exact_reference(model, batches), rtol=1e-4, atol=1e-7)


def test_resume_continues_the_same_data_order_and_refuses_another_objective(world, monkeypatch):
    seen, orig = [], L.DataCollatorForMLMDynamic.__call__

    def rec(self, examples):
        seen.append([tuple(ex["input_ids"]) for ex in examples])
        return orig(self, examples)

    monkeypatch.setattr(L.DataCollatorForMLMDynamic, "__call__", rec)
    full = L.CPTTrainer(make_cfg(world, world["root"] / "full"))
    plan = full.train_sampler.step_plan()
    full.train()
    whole = list(seen[:sum(plan)])
    ck = str(world["root"] / "full" / f"checkpoint_step_{SAVE_EVERY}")
    seen.clear()
    resumed = L.CPTTrainer(make_cfg(world, world["root"] / "resumed", resume_from=ck))
    resumed.train()
    assert seen[:sum(plan[SAVE_EVERY:])] == whole[sum(plan[:SAVE_EVERY]):]
    assert resumed.global_step == full.global_step == STEPS
    with pytest.raises(Exception, match="objective|loss_weighting"):
        L.CPTTrainer(make_cfg(world, world["root"] / "drift", resume_from=ck, loss_weighting="token",
                              corpus_preflight="off"))


# ------------------------------------------------------------------ refusals before any load
def test_token_exact_needs_the_token_budget_sampler_and_the_preflight(world):
    with pytest.raises(ValueError, match="token_budget > 0"):
        L.CPTTrainer(make_cfg(world, world["root"] / "x1", token_budget=0))
    with pytest.raises(ValueError, match="corpus_preflight 'required'"):
        L.CPTTrainer(make_cfg(world, world["root"] / "x2", corpus_preflight="off"))
    with pytest.raises(ValueError, match="unknown loss_weighting"):
        L.CPTTrainer(make_cfg(world, world["root"] / "x3", loss_weighting="token-exact"))


def test_a_corpus_that_fails_the_preflight_is_not_trained_on(world, tmp_path):
    import shutil
    broken = tmp_path / "v"
    shutil.copytree(world["version"], broken)
    (broken / "leakage-audit.json").unlink()
    with pytest.raises(RuntimeError, match="does not pass the corpus preflight"):
        L.CPTTrainer(make_cfg(world, tmp_path / "out", dataset_path=str(broken / "dataset")))
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"as_paired_i_val.parquet": "ee"}))        # not what the audit read
    with pytest.raises(RuntimeError, match="other benchmark files"):
        L.CPTTrainer(make_cfg(world, tmp_path / "out2", corpus_benchmark_sha256=str(bench)))
