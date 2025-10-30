# QUEST Repository Organization

This document describes the organization and purpose of each directory in the QUEST repository.

## Directory Structure

```
quest/
├── quest/              # Core package with models and training logic
├── parsers/            # Data parsing utilities for AIRR databases
├── scripts/            # Executable scripts organized by purpose
│   ├── training/       # Model training and evaluation scripts
│   ├── inference/      # Inference and prediction scripts
│   ├── data_processing/# Data parsing and preprocessing scripts
│   ├── analysis/       # Analysis and visualization scripts
│   └── utils/          # Helper scripts and shell scripts
├── tests/              # Unit and integration tests
├── notebooks/          # Jupyter notebooks for exploration
├── config/             # Configuration files (YAML, JSON)
├── env/                # Conda environment specifications
├── outputs/            # Generated outputs (gitignored)
│   ├── reports/        # Text reports
│   ├── csv/            # CSV analysis results
│   └── logs/           # Log files
├── data/               # Data directory (gitignored)
│   ├── raw/            # Raw input data
│   └── processed/      # Processed datasets
└── docs/               # Additional documentation
```

## Package Structure

### `quest/` - Core Package
The main Python package containing:
- **models/**: Neural network architectures (BERT, LSTM, Transformer)
- **config.py**: Configuration management
- **dataset.py**: Dataset classes
- **trainer.py**: Training utilities
- **utils.py**: Helper functions

### `parsers/` - Data Parsers
Utilities for parsing AIRR (Adaptive Immune Receptor Repertoire) database formats:
- AIRR bulk data parsers
- AIRR paired sequence parsers
- TCR stitching utilities
- Streaming parsers for large datasets

### `scripts/` - Executable Scripts

#### `scripts/training/`
- **ray_train.py**: Distributed training with Ray
- **ray_fine_tune.py**: Fine-tuning pre-trained models
- **ray_evaluator.py**: Model evaluation

#### `scripts/inference/`
- **run_inference.py**: Main inference script
- **inference_examples.py**: Example inference workflows
- **inference_with_dataset.py**: Batch inference on datasets
- **quick_inference.py**: Quick single-sequence inference

#### `scripts/data_processing/`
- **ray_datawriter.py**: Distributed data processing with Ray
- **reformat_data.py**: Data format conversion
- **run_parser_production.py**: Production data parsing
- **run_parser_test.py**: Test data parsing
- **reparse_databases.py**: Database re-parsing utilities

#### `scripts/analysis/`
- **analyze_duplication_parallel.py**: Parallel duplication analysis
- **calculate_perplexity.py**: Model perplexity calculation
- **evaluate_masked_predictions.py**: Masked prediction evaluation
- **export_embeddings.py**: Export model embeddings
- **best_worst_by_molecule.py**: Best/worst prediction analysis
- **summarize_data.py**: Data summarization utilities
- **create_source_mapping.py**: Source data mapping creation

#### `scripts/utils/`
- Shell scripts and utilities for setup and maintenance

### `tests/` - Tests
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Test utilities and fixtures

### `outputs/` - Generated Outputs
**Note**: This directory is gitignored. All generated files go here:
- **reports/**: Analysis reports and summaries
- **csv/**: CSV data files
- **logs/**: Execution logs

### `data/` - Data Directory
**Note**: This directory is gitignored (except for mappings):
- **raw/**: Original unprocessed data
- **processed/**: Tokenized and formatted datasets ready for training

## Usage

### Installation
```bash
# Install in development mode
pip install -e .

# Or install specific extras
pip install -e ".[dev]"
```

### Running Scripts
```bash
# Training
python scripts/training/ray_train.py --dataset data/processed/train

# Inference
python scripts/inference/run_inference.py --model checkpoints/best_model

# Analysis
python scripts/analysis/calculate_perplexity.py --dataset data/processed/test
```

### Entry Points
After installation, you can use command-line tools:
```bash
quest-train --dataset data/processed/train
quest-finetune --config config/finetune.json
quest-evaluate --model checkpoints/best_model
quest-inference --input sequences.fasta
```

## Best Practices

1. **Keep outputs separate**: All generated files go in `outputs/`
2. **Version control**: Only track source code and essential configs
3. **Data organization**: Raw data in `data/raw/`, processed in `data/processed/`
4. **Script organization**: Place scripts in appropriate subdirectories by function
5. **Testing**: Add tests for new functionality in `tests/`
6. **Documentation**: Update docs when adding new features

## Migration Notes

This structure was reorganized from a flat structure on October 30, 2025. If you have old scripts that reference files in the root directory, they have been moved to:
- Scripts → `scripts/{training,inference,data_processing,analysis}/`
- Outputs → `outputs/{reports,csv,logs}/`
- Tests → `tests/` or `tests/integration/`
