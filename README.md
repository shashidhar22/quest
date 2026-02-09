# Quantitative Understanding of Epitope Specificity in T-cells (QUEST)

## Environment Setup

### 1. Install Miniforge

Download and install [Miniforge](https://github.com/conda-forge/miniforge) (a minimal conda installer using conda-forge by default):

```bash
# Linux (x86_64)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3

# Initialize shell integration
$HOME/miniforge3/bin/mamba init
source ~/.bashrc
```

### 2. Create the conda environment

```bash
mamba env create -f env/environment.yaml
mamba activate quest
```

This installs Python 3.12, CUDA toolkit, pandas, and other system-level dependencies defined in `env/environment.yaml`.

### 3. Install Python requirements

```bash
pip install -r requirements.txt
```

For Flash Attention 2 (recommended for training, CUDA only):

```bash
pip install flash-attn --no-build-isolation
```

### 4. Install QUEST in editable mode

```bash
pip install -e .            # Core install
pip install -e ".[dev]"     # With dev tools (pytest, black, flake8, mypy)
```

---

## Data Processing

The data pipeline takes raw TCR/MHC sequencing data from `data/databases/`, standardizes the format, and deduplicates in a memory-efficient streaming fashion.

### Streaming Deduplication (`deduplicate_streaming.py`)

The primary data processing script. Runs a three-stage pipeline: molecule deduplication, permutation generation, and permutation deduplication.

```bash
# TRB sequences (CDR3-based)
python scripts/data_processing/deduplicate_streaming.py \
  --path data/databases/*/ \
  --output-deduplicated data/deduplicated/trb \
  --mode trb \
  --tmp-dir /mnt/ephemeral/temp

# TCR pairing (alpha-beta pairs)
python scripts/data_processing/deduplicate_streaming.py \
  --path data/databases/*/ \
  --output-deduplicated data/deduplicated/pairing \
  --mode tcr_pairing \
  --tmp-dir /mnt/ephemeral/temp

# Full-length TCR reconstruction with stitchr
python scripts/data_processing/deduplicate_streaming.py \
  --path data/databases/*/ \
  --output-deduplicated data/deduplicated/trb_full \
  --mode trb \
  --stitch-tcr \
  --tmp-dir /mnt/ephemeral/temp
```

**Key parameters:**
- `--path`: Input parquet directories (supports glob patterns)
- `--output-deduplicated`: Output directory for deduplicated parquet
- `--mode`: Processing mode (`tra`, `trb`, `tcr_pairing`, `mhc_binding`, `specificity`, `default`, `balanced`)
- `--tmp-dir`: Temp directory for intermediate files
- `--work-dir`: Working directory for large sorted chunks (defaults to `--tmp-dir`)
- `--stitch-tcr`: Generate full-length TCR sequences from CDR3 + gene segments
- `--no-permutations`: Skip permutation generation (just deduplicate)
- `--sort-chunk-size`: Chunk size for PyArrow sort (default: 200M lines)
- `--resume`: Resume from existing intermediate files
- `--sample`: Sample N files for testing

### Ray Data Writer (`ray_datawriter.py`)

Alternative deduplication pipeline using Ray for distributed processing.

```bash
python scripts/data_processing/ray_datawriter.py \
  --path data/databases/*/*.parquet \
  --output-deduplicated data/deduplicated/trb \
  --mode trb \
  --use-hash-dedup \
  --fast-mode
```

---

## Training

All trainers use PyTorch DDP for multi-GPU training and extend `BaseTCRTrainer` for hardware-agnostic support (CUDA and AWS Trainium). Launch multi-GPU training with `torchrun`.

### MLM Pre-training (`esm_native_trainer.py`)

The primary trainer. Native PyTorch DDP training for ESM2 masked language modeling with LoRA, Flash Attention 2, and sequence packing.

```bash
# Single GPU
python scripts/training/esm_native_trainer.py \
  --dataset_path data/deduplicated/trb \
  --output_dir ./output/esm2_mlm \
  --model_name facebook/esm2_t33_650M_UR50D \
  --batch_size 48 \
  --num_epochs 3 \
  --use_lora \
  --use_varlen \
  --report_to wandb

# Multi-GPU (8x)
torchrun --nproc_per_node=8 scripts/training/esm_native_trainer.py \
  --dataset_path data/deduplicated/trb \
  --output_dir ./output/esm2_mlm \
  --model_name facebook/esm2_t33_650M_UR50D \
  --batch_size 48 \
  --num_epochs 3 \
  --use_lora \
  --use_varlen
```

**Key parameters:**
- `--dataset_path`: Path to HuggingFace dataset (parquet)
- `--model_name`: ESM2 model (`facebook/esm2_t33_650M_UR50D`, `facebook/esm2_t36_3B_UR50D`, etc.)
- `--use_lora`: Apply LoRA to encoder (enabled by default)
- `--use_varlen`: Varlen packing with block-diagonal attention masking (recommended)
- `--use_length_bucketing`: Group similar-length sequences to minimize padding (default: true)
- `--mlm_probability`: Masking probability (default: 0.15)
- `--max_seq_length`: Maximum sequence length (default: 1024)
- `--resume_from_checkpoint`: Resume training from a checkpoint

### TCR Sequence Generation (`tcr_seq2seq_trainer.py`)

ESM2 encoder + Transformer decoder for conditional TCR sequence generation (SFT). Supports CDR-weighted loss, BLOSUM62 soft loss, and biological stopping criteria.

```bash
# Generate TCR beta from alpha
torchrun --nproc_per_node=8 scripts/training/tcr_seq2seq_trainer.py \
  --data_path data/deduplicated/pairing \
  --output_dir ./output/tcr_seq2seq \
  --task BETA \
  --encoder_checkpoint ./output/esm2_mlm/best_model.pt \
  --use_lora \
  --batch_size 8

# With CDR-weighted loss and BLOSUM soft loss
torchrun --nproc_per_node=8 scripts/training/tcr_seq2seq_trainer.py \
  --data_path data/deduplicated/pairing \
  --output_dir ./output/tcr_seq2seq_cdr \
  --task BETA \
  --use_lora \
  --use_cdr_weighting \
  --cdr3_weight 4.0 \
  --use_blosum_loss \
  --blosum_alpha 0.1
```

**Key parameters:**
- `--task`: Generation task (`ALPHA`, `BETA`, `PEPTIDE`)
- `--encoder_checkpoint`: Path to pre-trained encoder checkpoint (from MLM)
- `--use_cdr_weighting`: Weight loss by CDR region (requires ANARCI)
- `--use_blosum_loss`: Add BLOSUM62 soft loss
- `--use_biological_stopping`: Early stop on biological metrics instead of just loss
- `--decoder_layers`, `--decoder_heads`, `--decoder_dim`: Decoder architecture

### Peptide-to-MHC Generation (`peptide_mhc_seq2seq_trainer.py`)

Encoder-decoder for predicting MHC sequences from peptide input. Supports Class I (single chain) and Class II (two chains).

```bash
# Class I (single MHC chain, ~365 AA)
torchrun --nproc_per_node=8 scripts/training/peptide_mhc_seq2seq_trainer.py \
  --data_path data/deduplicated/mhc_class_i \
  --output_dir ./output/peptide_mhc_i \
  --task MHC_CLASS_I \
  --use_lora \
  --batch_size 8

# Class II (two MHC chains, ~520 AA)
torchrun --nproc_per_node=8 scripts/training/peptide_mhc_seq2seq_trainer.py \
  --data_path data/deduplicated/mhc_class_ii \
  --output_dir ./output/peptide_mhc_ii \
  --task MHC_CLASS_II \
  --max_decoder_length 550 \
  --batch_size 4
```

### Contrastive Learning (`tcr_robust_contrastive_trainer.py`)

TCR alpha-beta pairing with MoCo-style momentum encoder, hard negative mining, and full-corpus retrieval evaluation.

```bash
torchrun --nproc_per_node=4 scripts/training/tcr_robust_contrastive_trainer.py \
  --data_path data/deduplicated/pairing \
  --output_dir ./output/contrastive \
  --use_lora \
  --batch_size 64 \
  --temperature 0.07 \
  --num_distractors 1000 \
  --use_curriculum
```

**Key parameters:**
- `--temperature`: InfoNCE temperature (default: 0.07)
- `--num_distractors`: Negative samples per batch (default: 1000)
- `--use_curriculum`: Short-to-long curriculum learning (default: true)
- `--pretrained_checkpoint`: Initialize from pre-trained PEFT checkpoint

### Cross-Encoder (`tcr_cross_encoder_trainer.py`)

Binary classifier for TCR alpha-beta pairing using concatenated input with self-attention.

```bash
python scripts/training/tcr_cross_encoder_trainer.py \
  --data_path data/deduplicated/pairing \
  --output_dir ./output/cross_encoder \
  --use_lora \
  --batch_size 16 \
  --neg_ratio 3 \
  --pooling attention
```

### Common Training Options

All trainers share these options via `BaseTCRTrainer`:

| Option | Description |
|--------|-------------|
| `--backend auto/cuda/xla` | Hardware backend (auto-detects CUDA vs Trainium) |
| `--use_lora` | LoRA fine-tuning (with `--lora_r`, `--lora_alpha`, `--lora_dropout`) |
| `--gradient_checkpointing` | Reduce memory usage (default: enabled) |
| `--report_to wandb/none` | Experiment tracking |
| `--eval_steps N` | Evaluate every N optimizer steps |
| `--patience N` | Early stopping patience |
| `--overfit_check` | Sanity check: overfit a single batch |

---