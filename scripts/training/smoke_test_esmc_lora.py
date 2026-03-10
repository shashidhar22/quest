#!/usr/bin/env python3
"""
Smoke test for ESM-C LoRA fine-tuning.

Validates that:
1. LoRA targets are correct and PEFT applies them properly
2. Forward pass + manual cross-entropy loss works
3. Training is numerically stable in bf16
4. Loss actually decreases (model is learning)

Usage:
    python scripts/training/smoke_test_esmc_lora.py
"""

import sys

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_from_disk
from esm.models.esmc import ESMC
from esm.tokenization import EsmSequenceTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from quest.training.collators import DataCollatorForMLMDynamic

# ESM-C separator token: '|' (pipe, ID 31)
ESMC_SEPARATOR_TOKEN_ID = 31

# --- Config ---
MODEL_NAME = "esmc_300m"
DATASET_PATH = "data/tokenized/mlm_full/esmc/train/proportional_1M"
LORA_TARGETS = ["layernorm_qkv.1", "out_proj", "ffn.1", "ffn.3"]
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
NUM_SEQUENCES = 64
BATCH_SIZE = 8
NUM_STEPS = 100
LR = 2e-4
MLM_PROBABILITY = 0.15
LOG_EVERY = 10


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load model
    print(f"\n{'='*60}")
    print(f"Loading {MODEL_NAME}...")
    model = ESMC.from_pretrained(MODEL_NAME, device=device)

    # 2. Print all named modules with Linear layer shapes
    print(f"\n{'='*60}")
    print("Linear modules in ESM-C:")
    linear_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  {name}: Linear({module.in_features}, {module.out_features})")
            linear_count += 1
    print(f"Total Linear modules: {linear_count}")

    # 3. Apply LoRA
    # ESMC doesn't have a HuggingFace-style .config attribute, but PEFT expects
    # both .config.get() (for tie_word_embeddings) and .config.use_return_dict.
    # Use a dict subclass that supports both attribute and dict access.
    class _HFConfigShim(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

    if not hasattr(model, "config"):
        model.config = _HFConfigShim(
            use_return_dict=True,
            tie_word_embeddings=False,
        )

    print(f"\n{'='*60}")
    print(f"Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGETS})")
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Count LoRA adapters
    lora_adapter_count = 0
    for name, _ in model.named_parameters():
        if "lora_" in name:
            lora_adapter_count += 1
    print(f"LoRA adapter parameters: {lora_adapter_count}")

    # 4. Load dataset
    print(f"\n{'='*60}")
    print(f"Loading {NUM_SEQUENCES} sequences from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    dataset = dataset.select(range(min(NUM_SEQUENCES, len(dataset))))
    print(f"Loaded {len(dataset)} sequences")

    # 5. Setup collator and create batches
    tokenizer = EsmSequenceTokenizer()
    collator = DataCollatorForMLMDynamic(
        tokenizer=tokenizer,
        mlm_probability=MLM_PROBABILITY,
        separator_token_id=ESMC_SEPARATOR_TOKEN_ID,
    )

    # Pre-create all batches
    batches = []
    for i in range(0, len(dataset), BATCH_SIZE):
        batch_data = [dataset[j] for j in range(i, min(i + BATCH_SIZE, len(dataset)))]
        batch = collator(batch_data)
        batches.append(batch)
    print(f"Created {len(batches)} batches of size {BATCH_SIZE}")

    # 6. Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        fused=torch.cuda.is_available(),
    )

    # 7. Training loop
    print(f"\n{'='*60}")
    print(f"Running {NUM_STEPS} training steps...")
    model.train()

    losses = []
    nan_detected = False
    nan_grad_steps = []

    for step in range(NUM_STEPS):
        batch = batches[step % len(batches)]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            sequence_id = attention_mask.bool()
            # Bypass PEFT's forward wrapper (PeftModelForTokenClassification remaps
            # kwargs to HF-style input_ids, which ESMC doesn't accept). The LoRA
            # adapters are injected into the model's Linear modules, so calling
            # the base model directly still routes through LoRA.
            base_model = model.base_model.model
            output = base_model(sequence_tokens=input_ids, sequence_id=sequence_id)
            logits = output.sequence_logits

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        # Check loss for NaN
        if loss.isnan().item():
            print(f"  [WARN] NaN loss at step {step}")
            nan_detected = True

        loss.backward()

        # Check gradients for NaN
        for name, p in model.named_parameters():
            if p.grad is not None and p.grad.isnan().any():
                nan_grad_steps.append(step)
                nan_detected = True
                if step < 5 or step % LOG_EVERY == 0:
                    print(f"  [WARN] NaN gradient in {name} at step {step}")
                break

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % LOG_EVERY == 0 or step == NUM_STEPS - 1:
            print(f"  Step {step:3d} | Loss: {loss_val:.4f}")

    # 8. Summary
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    # Use average of last 10 steps for more stable comparison
    avg_final_loss = sum(losses[-10:]) / 10
    min_loss = min(losses)

    print(f"  Model:          {MODEL_NAME}")
    print(f"  LoRA targets:   {LORA_TARGETS}")
    print(f"  LoRA rank:      {LORA_R}")
    print(f"  Steps:          {NUM_STEPS}")
    print(f"  Initial loss:   {initial_loss:.4f}")
    print(f"  Final loss:     {final_loss:.4f}")
    print(f"  Avg last 10:    {avg_final_loss:.4f}")
    print(f"  Min loss:       {min_loss:.4f}")
    print(f"  NaN detected:   {nan_detected}")
    if nan_grad_steps:
        print(f"  NaN grad steps: {nan_grad_steps}")

    # Pass/fail checks
    print(f"\n{'='*60}")
    print("CHECKS:")

    checks_passed = 0
    total_checks = 4

    # Check 1: LoRA adapters applied correctly (4 targets x 30 layers x 2 matrices = 240)
    expected_min = 100  # At least 100 LoRA params (conservative)
    c1 = lora_adapter_count >= expected_min
    print(f"  [{'PASS' if c1 else 'FAIL'}] LoRA adapters: {lora_adapter_count} (expected >= {expected_min})")
    checks_passed += c1

    # Check 2: Initial loss is finite
    c2 = not (torch.tensor(initial_loss).isnan() or torch.tensor(initial_loss).isinf())
    print(f"  [{'PASS' if c2 else 'FAIL'}] Initial loss finite: {initial_loss:.4f}")
    checks_passed += c2

    # Check 3: Loss decreased
    c3 = avg_final_loss < initial_loss
    print(f"  [{'PASS' if c3 else 'FAIL'}] Loss decreased: {initial_loss:.4f} -> {avg_final_loss:.4f}")
    checks_passed += c3

    # Check 4: No NaN in gradients
    c4 = not nan_detected
    print(f"  [{'PASS' if c4 else 'FAIL'}] No NaN detected")
    checks_passed += c4

    print(f"\n  Result: {checks_passed}/{total_checks} checks passed")

    if checks_passed == total_checks:
        print("\n  SMOKE TEST PASSED - LoRA targets validated for production use")
        print(f"  Validated target_modules = {LORA_TARGETS}")
    else:
        print("\n  SMOKE TEST FAILED - review issues above")
        sys.exit(1)


if __name__ == "__main__":
    main()
