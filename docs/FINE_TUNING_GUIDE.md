# Fine-Tuning Guide

Comprehensive guide for fine-tuning protein language models using `scripts/training/fine_tune.py`.

## Table of Contents
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Fine-Tuning Modes](#fine-tuning-modes)
- [Example Commands](#example-commands)
- [Advanced Options](#advanced-options)
- [Memory Optimization](#memory-optimization)
- [Hardware Recommendations](#hardware-recommendations)

---

## Quick Start

### Basic ESM2 Fine-Tuning
```bash
python scripts/training/fine_tune.py \
  --dataset_path data/masked/phase_zero/database/1M/balanced/esm2_cdr_dmlm_1M_balanced \
  --mode mlm \
  --model_path facebook/esm2_t12_35M_UR50D \
  --output_dir checkpoints/esm2_35M_finetune \
  --num_epochs 3 \
  --batch_size 8 \
  --use_pre_masked \
  --fp16
```

### Basic ESM3 Fine-Tuning
```bash
python scripts/training/fine_tune.py \
  --dataset_path data/masked/phase_zero/database/1M/balanced/esm3_cdr_dmlm_1M_balanced \
  --mode mlm \
  --model_path esm3-small \
  --output_dir checkpoints/esm3_finetune \
  --num_epochs 3 \
  --batch_size 4 \
  --use_pre_masked \
  --use_lora \
  --fp16
```

---

## Supported Models

### ESM2 Models (HuggingFace)
| Model | Parameters | GPU Memory | Recommended Instance |
|-------|-----------|------------|---------------------|
| `facebook/esm2_t6_8M_UR50D` | 8M | ~2GB | Any |
| `facebook/esm2_t12_35M_UR50D` | 35M | ~4GB | g5.xlarge |
| `facebook/esm2_t30_150M_UR50D` | 150M | ~8GB | g5.xlarge |
| `facebook/esm2_t33_650M_UR50D` | 650M | ~12GB | g5.xlarge |
| `facebook/esm2_t36_3B_UR50D` | 3B | ~40GB+ | g5.12xlarge, p3.2xlarge |

### ESM3 Models (via esm package)
| Model | Parameters | GPU Memory | Recommended Instance |
|-------|-----------|------------|---------------------|
| `esm3-small` | 1.4B | ~40GB+ | g5.2xlarge, p3.2xlarge, A100 |

**Note:** ESM3 requires the `esm` package: `pip install esm`

---

## Fine-Tuning Modes

The `--mode` argument determines data filtering and masking strategy:

### 1. `mlm` - Masked Language Modeling
- **Description**: Standard 15% random masking for general protein language understanding
- **Use Case**: General pre-training or adaptation to new protein domains
- **Data**: Uses all sequences in the dataset

### 2. `tra` / `trb` - TCR Chain CDR3 Prediction
- **Description**: Masks middle 5 amino acids of CDR3 region
- **Use Case**: TCR alpha or beta chain CDR3 sequence prediction
- **Data**: Filters to only TRA or TRB sequences

### 3. `tra_trb_pairing` - TCR Pairing Prediction
- **Description**: Masks entire first chain to predict from second chain
- **Use Case**: Learning TRA-TRB pairing patterns
- **Data**: Requires sequences containing both TRA and TRB

### 4. `tcr_mhc` - TCR-MHC Binding
- **Description**: Masks first molecule (or both TCR chains if first two)
- **Use Case**: TCR-MHC interaction prediction
- **Data**: Requires TCR chain(s) + MHC molecule(s)

### 5. `peptide_mhc` - Peptide-MHC Binding
- **Description**: Masks first molecule (or both MHC chains if first two)
- **Use Case**: Peptide-MHC interaction prediction
- **Data**: Requires peptide + MHC molecule(s)

### 6. `specificity` - TCR Specificity Prediction
- **Description**: Masks first molecule in complete TCR complexes
- **Use Case**: Full TCR specificity (TCR + peptide + MHC)
- **Data**: Requires TCR chain(s) + peptide + MHC molecule(s)

---

## Example Commands

### Production ESM2-650M Fine-Tuning
```bash
python scripts/training/fine_tune.py \
  --dataset_path data/masked/phase_zero/database/1M/balanced/esm2_cdr_dmlm_1M_balanced \
  --mode mlm \
  --model_path facebook/esm2_t33_650M_UR50D \
  --output_dir checkpoints/esm2_650M_cdr_dmlm_1M_balanced \
  --num_epochs 3 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --learning_rate 5e-5 \
  --use_lora \
  --use_pre_masked \
  --fp16 \
  --wandb_project quest-finetuning-phase0 \
  --gradient_accumulation_steps 8 \
  --wandb_run_name esm2_650M_cdr_dmlm_1M_balanced \
  --seed 12357
```

**Key Features:**
- LoRA for parameter-efficient training (~1% trainable parameters)
- Gradient accumulation for effective batch size of 32 (4 × 8)
- Pre-masked dataset for faster training
- W&B logging for experiment tracking
- Reproducible with fixed seed

### Production ESM3-Small Fine-Tuning
```bash
python scripts/training/fine_tune.py \
  --dataset_path data/masked/phase_zero/database/1M/balanced/esm3_cdr_dmlm_1M_balanced \
  --mode mlm \
  --model_path esm3-small \
  --output_dir checkpoints/esm3_cdr_dmlm_1M_balanced \
  --num_epochs 3 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --learning_rate 5e-5 \
  --use_lora \
  --use_pre_masked \
  --fp16 \
  --wandb_project quest-finetuning-phase0 \
  --gradient_accumulation_steps 8 \
  --wandb_run_name esm3_cdr_dmlm_1M_balanced \
  --seed 12357 \
  --lora_r 8 \
  --lora_alpha 16
```

**ESM3-Specific:**
- Auto-detects ESM3 architecture and finds correct LoRA target modules
- Uses custom ESM3ForMaskedLM wrapper for HuggingFace Trainer compatibility
- Higher LoRA rank (8) for larger model capacity

### Task-Specific: TRA CDR3 Prediction
```bash
python scripts/training/fine_tune.py \
  --dataset_path data/masked/tcr_data \
  --mode tra \
  --model_path facebook/esm2_t12_35M_UR50D \
  --output_dir checkpoints/esm2_tra_cdr3 \
  --num_epochs 5 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --cdr3_mask_length 5 \
  --fp16
```

### Full Fine-Tuning (No LoRA)
```bash
python scripts/training/fine_tune.py \
  --dataset_path data/masked/phase_zero/database/1M/balanced/esm2_cdr_dmlm_1M_balanced \
  --mode mlm \
  --model_path facebook/esm2_t12_35M_UR50D \
  --output_dir checkpoints/esm2_35M_full_finetune \
  --num_epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --use_pre_masked \
  --fp16
```

### Test Mode (Quick Validation)
```bash
python scripts/training/fine_tune.py \
  --dataset_path data/masked/phase_zero/database/1M/balanced/esm2_cdr_dmlm_1M_balanced \
  --mode mlm \
  --model_path facebook/esm2_t12_35M_UR50D \
  --output_dir checkpoints/test_run \
  --num_epochs 1 \
  --batch_size 8 \
  --use_pre_masked \
  --test \
  --fp16
```

---

## Advanced Options

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_path` | str | **required** | Path to HuggingFace dataset directory |
| `--mode` | str | **required** | Fine-tuning mode (see [Modes](#fine-tuning-modes)) |
| `--model_path` | str | **required** | HuggingFace model name or ESM3 model identifier |
| `--output_dir` | str | **required** | Directory to save checkpoints and final model |

### Training Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_epochs` | int | 3 | Number of training epochs |
| `--batch_size` | int | 32 | Training batch size per device |
| `--eval_batch_size` | int | `batch_size` | Evaluation batch size per device |
| `--learning_rate` | float | 5e-5 | Learning rate |
| `--warmup_steps` | int | 500 | Number of warmup steps |
| `--weight_decay` | float | 0.01 | Weight decay for optimizer |
| `--max_grad_norm` | float | 1.0 | Max gradient norm for clipping |
| `--gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |

### LoRA (Parameter-Efficient Fine-Tuning)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_lora` | flag | False | Enable LoRA |
| `--lora_r` | int | 16 | LoRA rank (lower = fewer parameters) |
| `--lora_alpha` | int | 32 | LoRA alpha (scaling factor) |
| `--lora_dropout` | float | 0.05 | LoRA dropout rate |
| `--lora_target_modules` | list | `["query", "value"]` | Target modules for LoRA |

**LoRA Recommendations:**
- ESM2: Default `["query", "value"]` works well
- ESM3: Auto-detects correct modules (attention + FFN layers)
- Lower `lora_r` (4-8) for memory-constrained setups
- Higher `lora_r` (16-32) for better performance on large datasets

### Masking Strategy

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_pre_masked` | flag | False | Use pre-masked dataset (much faster!) |
| `--mlm_probability` | float | 0.15 | Probability for MLM masking (if not pre-masked) |
| `--cdr3_mask_length` | int | 5 | Amino acids to mask in CDR3 (for `tra`/`trb` modes) |

### Logging & Evaluation

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--wandb_project` | str | None | W&B project name |
| `--wandb_run_name` | str | None | W&B run name |
| `--logging_steps` | int | 100 | Log every N steps |
| `--eval_steps` | int | None | Evaluate every N steps (default: per epoch) |
| `--save_steps` | int | None | Save checkpoint every N steps (default: per epoch) |
| `--max_eval_samples` | int | None | Limit validation set size |
| `--log_prediction_examples` | flag | False | Log prediction examples to W&B |

### Optimization & Precision

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fp16` | flag | False | Use FP16 mixed precision |
| `--bf16` | flag | False | Use BF16 mixed precision |
| `--gradient_checkpointing` | flag | False | Enable gradient checkpointing (saves ~30% memory) |
| `--optim` | str | `adamw_torch` | Optimizer (`adamw_8bit` for memory savings) |
| `--max_seq_length` | int | None | Maximum sequence length (truncate longer) |

### Other

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--test` | flag | False | Test mode (1000 train, 200 val samples) |
| `--seed` | int | 42 | Random seed for reproducibility |

---

## Memory Optimization

### Strategies for Limited GPU Memory

The script automatically detects GPU memory and provides recommendations. Manual optimization strategies:

#### 1. **Use LoRA** (Recommended)
Reduces trainable parameters by ~99%
```bash
--use_lora --lora_r 8 --lora_alpha 16
```

#### 2. **Enable Gradient Checkpointing**
Saves ~30% memory at ~20% speed cost
```bash
--gradient_checkpointing
```

#### 3. **Use 8-bit Optimizer**
Saves ~50% optimizer memory
```bash
--optim adamw_8bit  # Requires: pip install bitsandbytes
```

#### 4. **Reduce Batch Size + Increase Gradient Accumulation**
Maintains effective batch size while reducing memory
```bash
--batch_size 1 --gradient_accumulation_steps 32
```

#### 5. **Truncate Sequences**
Limits maximum sequence length
```bash
--max_seq_length 512
```

#### 6. **Reduce Evaluation Set**
Limits validation set size during evaluation
```bash
--max_eval_samples 10000
```

### Memory-Optimized Command (24GB GPU)
```bash
python scripts/training/fine_tune.py \
  --dataset_path data/masked/phase_zero/database/1M/balanced/esm2_cdr_dmlm_1M_balanced \
  --mode mlm \
  --model_path facebook/esm2_t33_650M_UR50D \
  --output_dir checkpoints/esm2_650M_mem_optimized \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --eval_batch_size 1 \
  --num_epochs 3 \
  --use_lora \
  --lora_r 4 \
  --use_pre_masked \
  --gradient_checkpointing \
  --optim adamw_8bit \
  --max_seq_length 512 \
  --fp16
```

---

## Hardware Recommendations

### GPU Requirements by Model

| Model | Min GPU | Recommended GPU | Batch Size | Notes |
|-------|---------|-----------------|------------|-------|
| ESM2-8M | 4GB | 8GB | 32-64 | Any modern GPU |
| ESM2-35M | 8GB | 16GB | 16-32 | T4, RTX 3060 |
| ESM2-150M | 16GB | 24GB | 8-16 | RTX 3090, A10 |
| ESM2-650M | 24GB | 32GB | 4-8 | A100-40GB, g5.xlarge |
| ESM2-3B | 40GB+ | 80GB | 1-2 | A100-80GB, g5.12xlarge |
| ESM3-small | 40GB+ | 80GB | 2-4 | A100-80GB, p3.2xlarge |

### AWS Instance Recommendations

| Instance | GPU | GPU Memory | Suitable Models | Hourly Cost* |
|----------|-----|------------|-----------------|--------------|
| g5.xlarge | A10G | 24GB | ESM2 ≤650M | ~$1.00 |
| g5.2xlarge | A10G | 24GB | ESM2 ≤650M | ~$1.20 |
| g5.12xlarge | 4× A10G | 96GB | ESM2-3B, ESM3 | ~$5.00 |
| p3.2xlarge | V100 | 16GB | ESM2 ≤150M | ~$3.00 |
| p3.8xlarge | 4× V100 | 64GB | ESM2-3B | ~$12.00 |
| p4d.24xlarge | 8× A100 | 320GB | Any model | ~$32.00 |

*Approximate on-demand pricing (2025)

### Performance Tips

1. **Pre-mask datasets** using `scripts/data_processing/pre_mask_dataset.py` for 2-3× faster training
2. **Use mixed precision** (`--fp16` or `--bf16`) for 2× speedup and 40% memory savings
3. **Gradient accumulation** maintains large effective batch sizes on small GPUs
4. **LoRA** enables fine-tuning billion-parameter models on consumer GPUs
5. **W&B logging** tracks experiments without performance overhead

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:** `CUDA out of memory` error during training

**Solutions:**
1. Reduce `--batch_size` to 1
2. Increase `--gradient_accumulation_steps`
3. Add `--gradient_checkpointing`
4. Use `--optim adamw_8bit`
5. Add `--max_seq_length 512`
6. Enable `--use_lora` with lower `--lora_r`

### Slow Training

**Symptoms:** Very low iterations/second

**Solutions:**
1. Use `--use_pre_masked` (requires pre-masked dataset)
2. Remove `--gradient_checkpointing` if memory allows
3. Reduce `--max_eval_samples` to speed up validation
4. Increase `--dataloader_num_workers` (edit in code)

### LoRA Target Module Errors

**Symptoms:** `Target modules not found` error

**Solutions:**
- For ESM2: Use `--lora_target_modules query value`
- For ESM3: Script auto-detects; check console output for detected modules
- Manual override: `--lora_target_modules <module1> <module2>`

---

## Pre-Processing Datasets

### Create Pre-Masked Dataset

Pre-masking datasets speeds up training 2-3×:

```bash
python scripts/data_processing/pre_mask_dataset.py \
  --input_dataset data/tokenized/my_dataset \
  --output_dataset data/masked/my_dataset_masked \
  --mode mlm \
  --mlm_probability 0.15
```

Then use `--use_pre_masked` when training.

---

## Best Practices

### For Production Training

1. **Use W&B**: Track all experiments with `--wandb_project` and `--wandb_run_name`
2. **Fix seeds**: Use `--seed 12357` for reproducibility
3. **Pre-mask data**: Create masked datasets offline for faster iteration
4. **Start small**: Test with `--test` flag before full runs
5. **Use LoRA**: Unless you specifically need full fine-tuning
6. **Monitor memory**: Check GPU utilization with `nvidia-smi`
7. **Save often**: Use `--save_steps` for long training runs

### For Hyperparameter Tuning

```bash
# Example sweep over learning rates
for lr in 1e-5 5e-5 1e-4; do
  python scripts/training/fine_tune.py \
    --dataset_path data/masked/phase_zero/database/1M/balanced/esm2_cdr_dmlm_1M_balanced \
    --mode mlm \
    --model_path facebook/esm2_t12_35M_UR50D \
    --output_dir checkpoints/lr_sweep_${lr} \
    --learning_rate ${lr} \
    --use_pre_masked \
    --use_lora \
    --fp16 \
    --wandb_project quest-hp-sweep \
    --wandb_run_name lr_${lr}
done
```

---

## Questions?

See the main project README or open an issue on GitHub.
