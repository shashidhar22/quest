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

## Agent Team Roles for Data Curation

This section defines teammate roles for Claude Code Agent Teams. Agent teams allow a team lead to spawn independent Claude Code instances ("teammates") that share a task list and can message each other.

### Data Curation Pipeline Overview

The 5-stage curation flow:

1. **Standardize** — Convert raw database exports to 15-column target schema
   - Scripts: `scripts/data_processing/standardize/*.py`
   - Core: `quest/data/standardization.py` (`standardize_dataframe()`)
2. **Deduplicate** — 3-stage streaming dedup (exact hash, CDR3 near-dedup, cross-source)
   - Script: `scripts/data_processing/deduplicate_streaming.py`
3. **Validate** — Enforce quality rules, track dropped records
   - Core: `quest/data/standardization.py` (validation in `standardize_dataframe()`)
4. **Create datasets** — Stratified train/val/test splits with leakage checks
   - Scripts: `scripts/data_processing/create_tcr_*.py`
5. **Tokenize** — Convert sequences to model-ready token IDs
   - Script: `scripts/data_processing/tokenize_sequences.py`

### Target Schema

The 15 `TARGET_COLUMNS` from `quest/data/standardization.py`:

| Column | Description |
|--------|-------------|
| `tra` | TCR alpha chain CDR3 amino acid sequence |
| `trav_gene` | TCR alpha V gene |
| `trad_gene` | TCR alpha D gene |
| `traj_gene` | TCR alpha J gene |
| `trb` | TCR beta chain CDR3 amino acid sequence |
| `trbv_gene` | TCR beta V gene |
| `trbd_gene` | TCR beta D gene |
| `trbj_gene` | TCR beta J gene |
| `peptide` | Peptide/epitope sequence |
| `mhc_one` | MHC class I allele |
| `mhc_two` | MHC class II allele |
| `binding` | Binding label (positive/negative) |
| `score` | Binding score (continuous) |
| `source` | Source database name |
| `study_id` | Study identifier |

### Quality Rules

- **CDR3 length**: CDR3 sequences must be >= 4 amino acids
- **Gene naming**: V/D/J genes must start with "TR" (e.g., TRBV, TRAV)
- **MHC alleles**: Must be human HLA alleles (validated via tidytcells when available)
- **Dropped records**: All dropped records must be tracked with reason, source_file, row_index, field, and raw_value in a dropped_df DataFrame

### Teammate Role Definitions

#### 1. Standardization Specialist

**Scope**: Adding new database standardizers, fixing column mappings and normalization logic.

**Key files**:
- `scripts/data_processing/standardize/*.py` — Per-database standardizer scripts
- `quest/data/standardization.py` — Core `standardize_dataframe()`, `TARGET_COLUMNS`, validation

**Responsibilities**:
- Implement new standardizer scripts following the pattern in existing standardizers
- Map database-specific column names to `TARGET_COLUMNS`
- Ensure gene names are normalized (IMGT format via tidytcells)
- Add source and study_id metadata

#### 2. Data Quality Auditor

**Scope**: Validation logic, test coverage, dropped record analysis.

**Key files**:
- `quest/data/standardization.py` — Validation rules in `standardize_dataframe()`
- `tests/test_standardization.py` — Tests for core standardization logic
- `tests/test_standardizers.py` — Tests for individual database standardizers

**Responsibilities**:
- Review and strengthen validation rules
- Ensure all edge cases have test coverage
- Analyze dropped_df outputs to identify systematic data quality issues
- Verify that quality rules (CDR3 >= 4 AA, gene naming, HLA alleles) are enforced

#### 3. Dedup Optimizer

**Scope**: Streaming deduplication pipeline tuning, TCR stitching.

**Key files**:
- `scripts/data_processing/deduplicate_streaming.py` — 3-stage streaming dedup
- `quest/parsers/tcr_stitcher.py` — Full-length TCR reconstruction from fragments

**Responsibilities**:
- Tune dedup thresholds and hash strategies
- Optimize memory usage for large-scale streaming dedup
- Verify cross-source dedup removes true duplicates without losing unique records
- Ensure TCR stitcher correctly reconstructs full-length sequences

#### 4. Dataset Validator

**Scope**: Stratified splits, epitope leakage checks, tokenization correctness.

**Key files**:
- `scripts/data_processing/create_tcr_*.py` — Dataset creation with stratified splits
- `scripts/data_processing/tokenize_sequences.py` — Sequence tokenization

**Responsibilities**:
- Verify train/val/test splits have no epitope leakage
- Check class balance across splits
- Validate tokenization round-trips (encode -> decode = original)
- Ensure special tokens (`[TRA]`, `[TRB]`, etc.) are correctly applied

#### 5. Input Data Curator

**Scope**: Reviewing scientific literature to understand the origin and experimental validity of new and existing input data sources. Making integration decisions based on evidence quality. Re-evaluating existing sources when new publications or quality concerns arise. Consulting the team when uncertain.

**Key files** (read-only — curator does NOT write code):
- `scripts/data_processing/standardize/_base.py` — BaseStandardizer pattern to understand
- `scripts/data_processing/standardize/*.py` — Existing standardizers as reference for how each database is handled
- `scripts/analysis/*_summary_export.py` — Per-database analysis scripts showing data completeness/quality patterns
- `quest/data/standardization.py` — `TARGET_COLUMNS`, normalization rules, `binding`/`score` field semantics

**Tools**: PubMed (search_articles, get_article_metadata, get_full_text_article), bioRxiv (search_preprints, get_preprint), WebSearch/WebFetch for database documentation pages

**Access model**: Read-only + decisions. The curator researches data provenance and produces integration decisions (evidence type, paired vs. unpaired, score thresholds). The Standardization Specialist then implements those decisions in code. This keeps scientific judgment and engineering cleanly separated.

**Responsibilities**:
1. Research the original publication(s) behind a new data source using PubMed/bioRxiv
2. Determine experimental method (tetramer staining, MIRA, yeast display, computational prediction, bulk sequencing, etc.)
3. Apply the **integration decision framework** (see below) to decide how records enter the pipeline
4. Document the integration decision as a task with: source name, publication DOI, experimental method, evidence type, recommended strategy, and any caveats
5. When unsure, create a task for team discussion rather than deciding unilaterally
6. Periodically review existing integrated sources to verify their evidence classification remains accurate and update integration decisions if new publications or quality assessments warrant changes

**Integration Decision Framework** — the key logic this agent applies:

| Evidence type | Integration strategy |
|--------------|---------------------|
| Experimentally validated TCR-pMHC binding (assay, tetramer, MIRA) | Expand each record into individual chains (tra, trb) and all multi-molecule interaction permutations (e.g., tra_trb, tra_peptide, tra_trb_peptide_mhc_one) as separate training records; all records used for MLM, interactions additionally used for contrastive/cross-encoder/seq2seq training |
| Quality-scored with threshold (e.g., VDJdb score) | Records >= threshold: expand into individual chains and all interaction permutations (same as experimentally validated, all used for MLM and interaction training); lower scores: use individual chains for MLM only |
| Experimental pMHC binding data, no TCR (e.g., NetMHCpan training data, IEDB mhc_bind/mhc_ligand) | Use for peptide-MHC modeling only (pMHC seq2seq, contrastive pMHC learning); do NOT use for TCR interaction training |
| Bulk repertoire (TCR-only, no epitope) | Use for MLM and TCR-only training modes (tra, trb, full_tra, full_trb) |
| Mixed source with labeled subsets | Split by label; apply appropriate strategy to each subset |
| Multi-source aggregation with experimentally validated data (e.g., TRAIT) | Same as experimentally validated; verify cross-source dedup catches overlap with individually integrated sources |

### Example Team Spawning Prompts

**Adding a new database**:
> "Create a team with 2 teammates: one standardization specialist to write a standardizer for [DATABASE_NAME], and one data quality auditor to write tests and validate the output."

**Full pipeline re-run**:
> "Create a team with 4 teammates: standardization specialist, dedup optimizer, data quality auditor, and dataset validator. Run the full pipeline on the latest data. Standardization blocks dedup, dedup blocks dataset creation, auditor reviews throughout."

**Integrating a new database**:
> "Create a team with 3 teammates: one input data curator to research the data provenance and decide integration strategy, one standardization specialist to write the standardizer, and one data quality auditor to validate the output."

**Fixing a normalization bug**:
> "Create a team with 1 teammate: a data quality auditor to investigate why [GENE_NAME] records are being dropped and fix the validation logic."

## NeuronX Distributed Training

### Tensor Parallelism on Trainium

QUEST supports tensor parallelism (TP) for training large ESM2 models across multiple Trainium chips on trn1.32xlarge instances using NeuronX Distributed.

**Configuration** (`quest/config.py` — `NEURON_TP_CONFIG`):
- `tensor_parallel_size`: Number of chips per TP group (default: 1, no TP)
- `pipeline_parallel_size`: Pipeline parallelism degree (default: 1)
- `zero_1`: Enable ZeRO-1 optimizer sharding (default: False)
- `sequence_parallel`: Enable sequence parallelism (default: False)

**ESM2 compatibility**: ESM2 t33_650M_UR50D has 20 attention heads. Compatible TP degrees without padding: 1, 2, 4, 5, 10, 20. For TP=8 (natural fit for half of trn1.32xlarge), heads are auto-padded to 24.

**Recommended configurations for trn1.32xlarge** (16 chips):
- TP=2, DP=8 — Best throughput for ESM2 650M
- TP=4, DP=4 — Balanced for larger batch sizes
- TP=8, DP=2 — For memory-constrained models

**Key files**:
- `quest/models/tensor_parallel.py` — `apply_tensor_parallelism()`, `pad_model()`, `get_tp_loss_fn()`
- `quest/training/backends/neuron_backend.py` — TP-aware `NeuronBackend`
- `scripts/training/launch_sagemaker_training.py` — `--tensor-parallel-size` CLI arg
