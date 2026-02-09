# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QUEST (Quantitative Understanding of Epitope Specificity in T-cells) is a machine learning framework for modeling TCR (T-cell receptor) and MHC (major histocompatibility complex) interactions. It uses protein language models (ESM2, ProtBERT) and custom neural architectures (LSTM, BiLSTM, Transformer) for tasks like masked language modeling, sequence generation, and contrastive learning on immune receptor sequences.

## Common Commands

### Installation
```bash
pip install -e .            # Editable install
pip install -e ".[dev]"     # With dev tools (pytest, black, flake8, mypy)
```

### Testing
```bash
pytest tests/
pytest tests/test_reformatter.py -v    # Single test file
pytest tests/ --cov=quest              # With coverage
```

### Formatting & Linting
```bash
black quest/ scripts/
flake8 quest/ scripts/
mypy quest/ scripts/
```

### Training (CLI entry points after install)
```bash
quest-train         # scripts.training.ray_train:main
quest-finetune      # scripts.training.ray_fine_tune:main
quest-evaluate      # scripts.training.ray_evaluator:main
quest-inference     # scripts.inference.run_inference:main
```

### Multi-GPU Training
```bash
torchrun --nproc_per_node=4 scripts/training/ray_train.py \
  --dataset /path/to/dataset --model lstm --num-epochs 10 --batch-size 128
```

## Architecture

### Two main packages

- **`quest/`** — Core library: models, metrics, data utilities, training infrastructure, parsers, constants
- **`scripts/`** — Executable scripts organized by purpose: `training/`, `data_processing/`, `inference/`, `benchmark/`, `analysis/`, `mil/`

### Core library (`quest/`)

#### `quest/models/`
Reusable model components and architectures:
- **positional_encoding.py** — Sinusoidal `PositionalEncoding`
- **attention_pooling.py** — `AttentionPooling`, `DeepProjectionHead`
- **lora_utils.py** — `apply_lora_to_encoder`, `load_encoder_checkpoint`
- **seq2seq_decoder.py** — `SelfAttnDropoutDecoder` with cross-attention forcing
- **seq2seq_model.py** — `TCRSeq2SeqModel`, `BLOSUM62SoftLoss`
- **pmhc_seq2seq_model.py** — `PeptideMHCSeq2SeqModel` for peptide-MHC generation
- **cross_encoder.py** — `TCRCrossEncoder`, `CrossEncoderBCELoss`
- **dual_encoder.py** — `TCRDualEncoder`, `MomentumEncoder`, `DistractorManager`
- **contrastive_loss.py** — `DistributedRobustInfoNCE`, `gather_embeddings`
- **factory.py** — Legacy model factory (`lstm`, `bilstm`, `transformer`)

#### `quest/metrics/`
Consolidated evaluation metrics:
- **sequence_metrics.py** — Levenshtein distance, BLOSUM62 similarity, sequence identity, `SequenceSimilarityCalculator`
- **biophysical.py** — Hydrophobicity, net charge, amino acid distribution, Jensen-Shannon divergence
- **position_wise.py** — `PositionWiseAnalyzer` for per-position accuracy tracking
- **cdr_analysis.py** — CDR annotation via ANARCI, `CDRAnnotationCache`, CDR weight presets
- **retrieval_metrics.py** — AUC, Recall@K, bootstrap CI, per-epitope breakdown

#### `quest/data/`
Shared dataset and generation utilities:
- **datasets.py** — `TCRSeq2SeqDataset`, `Seq2SeqCollator`, `TCRPairingDataset`, `ContrastivePairCollator`, `PeptideMHCDataset`
- **permutation_utils.py** — `GenerationTask` enum, permutation key utilities
- **generation_utils.py** — `greedy_decode`, `beam_search`, `sample_with_temperature`

#### `quest/training/`
Training infrastructure:
- **base_trainer.py** — `BaseTCRTrainer` abstract base class for all specialized trainers
- **callbacks.py** — `EarlyStopping`, `StreamingMetricsTracker`
- **collators.py** — `TaskSpecificMaskingCollator`, `DataCollatorForMLMWithPacking`, `DataCollatorForMLMDynamic`, `DataCollatorForMLMWithVarlen`, `ContrastivePairCollator`
- **samplers.py** — `LengthBucketSampler`, `DistributedLengthBucketSampler`, `CurriculumSampler`, `DistributedCurriculumSampler`
- **backends/** — Hardware-agnostic training: `AcceleratorBackend`, `CUDABackend`, `NeuronBackend`

#### `quest/parsers/`
TCR data parsers (moved from top-level `parsers/`):
- **streaming_parser.py** — `StreamingParserComplete` for processing large TCR datasets
- **tcr_stitcher.py** — `TCRStitcher` for full-length TCR reconstruction
- **cdr_region_identifier.py** — CDR region identification
- **format_to_imgt.py** — IMGT format standardization
- **streaming_format_mapper.py** — Format detection and mapping
- **airr/** — AIRR format parsers: `DatabaseParser`, `BulkFileParser`, `MiscFileParser`, `PairedFileParser`

### Training scripts (`scripts/training/`)

Specialized trainers that extend `BaseTCRTrainer`:
- **esm_native_trainer.py** — Native PyTorch ESM2 MLM fine-tuning (primary trainer)
- **tcr_seq2seq_trainer.py** — ESM2 encoder + Transformer decoder for TCR generation
- **peptide_mhc_seq2seq_trainer.py** — Seq2Seq for peptide-MHC generation
- **tcr_robust_contrastive_trainer.py** — Contrastive learning with momentum encoder
- **tcr_cross_encoder_trainer.py** — Cross-encoder for TCR-pMHC interaction

### Data pipeline

Raw TCR data (AIRR/Parquet/CSV) → `quest.parsers` (streaming parsers, standardization) → tokenization (`scripts/data_processing/ray_datawriter.py`) → HuggingFace Dataset → training

### Distributed training stack

- PyTorch DDP with NCCL backend for multi-GPU
- Ray for distributed data processing and hyperparameter tuning
- HuggingFace Accelerate + DeepSpeed for large model training
- AWS SageMaker integration (`launch_sagemaker_training.py`)
- WandB for experiment tracking

### Special token conventions (`quest/constants.py`)

Sequences use delimiter tokens: `[TRA]`/`[ETRA]` for TCR alpha, `[TRB]`/`[ETRB]` for TCR beta, `[PEP]`/`[EPEP]` for peptide, `[MHO]`/`[EMHO]` for MHC-I, `[MHT]`/`[EMHT]` for MHC-II. BPE tokenizer uses `[PAD]`, `[UNK]`, `[END]`.

## Key Configuration

- **`quest/config.py`** — Default hyperparameters per model type and Bayesian sweep configs
- **`env/environment.yaml`** — Conda environment (Python 3.12.6)
- **`scripts/training/accelerate_config_*.yaml`** — Accelerate/DeepSpeed configs for distributed training
