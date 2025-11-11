# Tokenization Guide

## Overview

The `tokenize_sequences.py` script converts deduplicated parquet files into tokenized HuggingFace datasets ready for training. It supports multiple model types and includes advanced sampling strategies for balanced datasets.

## Basic Usage

```bash
python scripts/data_processing/tokenize_sequences.py \
  --input-dir <path/to/deduplicated/parquet> \
  --output-dir <path/to/output> \
  --model-type <model_type> \
  --max-length <max_tokens> \
  --num-workers <num_processes>
```

## Model Types

### HuggingFace Pre-trained Models

| Model Type | Description | Tokenization | Vocab Size |
|------------|-------------|--------------|------------|
| `protbert` | ProtBERT (BERT for proteins) | Space-separated amino acids | 30 |
| `bert` | Standard BERT | Space-separated amino acids | 30 |
| `esm2` | ESM-2 (all sizes share tokenizer) | Direct amino acid encoding | 33 |
| `esm3` | ESM-3 (next-gen ESM) | Direct amino acid encoding | 33 |

### Custom Tokenizers

| Model Type | Description | Special Tokens | Vocab Size |
|------------|-------------|----------------|------------|
| `bpe` | Byte Pair Encoding | Molecule boundaries ([TRA], [TRB], etc.) | Configurable |
| `lstm` | BPE for LSTM models | Same as BPE | Configurable |
| `transformer` | BPE for Transformer models | Same as BPE | Configurable |

## Command-Line Arguments

### Required Arguments

- `--input-dir`: Directory containing deduplicated parquet files
- `--output-dir`: Directory to save tokenized dataset
- `--model-type`: Model type (see table above)

### Common Arguments

- `--max-length`: Maximum sequence length in tokens (default: 512)
  - **ProtBERT/BERT**: 512 tokens
  - **ESM-2**: 1024 tokens (can handle longer sequences)
  - **Custom BPE**: 512-2048 tokens

- `--num-workers`: Number of parallel workers for tokenization (default: 16)
  - Recommended: 50-60 for high-memory machines
  - Memory usage: ~2-4GB per worker

### Sampling Arguments

Control how much data to use and sampling strategy:

- `--sample`: Number of sequences to sample (optional)
  - Example: `--sample 1000000` = 1M sequences
  - If omitted, uses all data

- `--sample-mode`: Sampling strategy (requires `--sample`)
  - `proportional`: Maintains original permutation distribution
  - `balanced`: Equal samples per permutation type
  - If omitted with `--sample`, takes first N sequences

### Performance Arguments

- `--chunk-size`: Rows per processing chunk (default: 10,000,000)
  - Controls memory usage: chunk_size × 150 bytes ≈ RAM per chunk
  - Default 10M = ~1.5GB per chunk
  - Max recommended: 40M for 600GB RAM machines

- `--num-workers`: Parallel tokenization workers (default: 16)
  - Recommended range: 8-60 depending on CPU cores
  - More workers = faster, but more memory

### Data Split Arguments

- `--test-split`: Test set fraction (default: 0.1)
- `--val-split`: Validation set fraction (default: 0.1)
- Remaining data goes to training set (typically 0.8)

### BPE-Specific Arguments

For custom BPE tokenizers only:

- `--vocab-size`: BPE vocabulary size (default: 1000)
  - Recommended: 1000-5000 for LSTM
  - Recommended: 2000-10000 for Transformer

- `--train-bpe`: Train new BPE tokenizer (required for first run)
  - Trains on up to 1M sequences from training set
  - Saves tokenizer to `{output-dir}/tokenizer/`

## Example Commands

### 1. ESM-2 with Balanced Sampling (1M sequences)

```bash
python scripts/data_processing/tokenize_sequences.py \
  --input-dir path/to/deduplicated \
  --output-dir path/to/output/esm2_1M_balanced \
  --model-type esm2 \
  --max-length 1024 \
  --sample 1000000 \
  --sample-mode balanced \
  --chunk-size 10000000 \
  --num-workers 50
```

**What this does:**
- Samples 1M sequences with equal representation per permutation type
- Uses ESM-2 tokenizer (direct amino acid encoding)
- Processes in 10M row chunks
- Uses 50 parallel workers for fast tokenization
- Max sequence length: 1024 tokens

### 2. ProtBERT Full Dataset (No Sampling)

```bash
python scripts/data_processing/tokenize_sequences.py \
  --input-dir path/to/deduplicated \
  --output-dir path/to/output/protbert_full \
  --model-type protbert \
  --max-length 512 \
  --num-workers 60
```

**What this does:**
- Processes entire dataset (no sampling)
- Uses ProtBERT tokenizer (space-separated amino acids)
- Max sequence length: 512 tokens
- Uses 60 parallel workers

### 3. Custom BPE for Transformer (First Run)

```bash
python scripts/data_processing/tokenize_sequences.py \
  --input-dir path/to/deduplicated \
  --output-dir path/to/output/transformer_bpe \
  --model-type transformer \
  --max-length 1024 \
  --vocab-size 5000 \
  --train-bpe \
  --num-workers 50
```

**What this does:**
- Trains new BPE tokenizer with 5000 vocab size
- Saves tokenizer to `{output-dir}/tokenizer/`
- Tokenizes entire dataset
- Max sequence length: 1024 tokens

### 4. Proportional Sampling (Maintains Distribution)

```bash
python scripts/data_processing/tokenize_sequences.py \
  --input-dir path/to/deduplicated \
  --output-dir path/to/output/esm2_10M_proportional \
  --model-type esm2 \
  --max-length 1024 \
  --sample 10000000 \
  --sample-mode proportional \
  --num-workers 50
```

**What this does:**
- Samples 10M sequences maintaining original permutation distribution
- If original has 60% TRA, 20% TRB, 20% peptide, sample maintains this ratio

## Output Structure

After tokenization, the output directory contains:

```
output-dir/
├── hf_dataset/                    # HuggingFace DatasetDict (main output)
│   ├── train/
│   │   └── data-*.arrow           # Training data in Arrow format
│   ├── validation/
│   │   └── data-*.arrow           # Validation data
│   ├── test/
│   │   └── data-*.arrow           # Test data
│   └── dataset_dict.json          # Dataset metadata
│
├── tokenizer/                     # Saved tokenizer
│   ├── tokenizer_config.json     # Tokenizer configuration
│   ├── special_tokens_map.json   # Special tokens
│   └── vocab.txt (or .json)      # Vocabulary
│
└── train/ validation/ test/       # Parquet backups (compatibility)
    └── *.parquet                  # Same data in Parquet format
```

## Dataset Schema

Each dataset contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `input_ids` | List[int] | Token IDs for the sequence |
| `attention_mask` | List[int] | Attention mask (1=real token, 0=padding) |
| `sequence` | str | Original concatenated sequence |
| `permutation_key` | str | Permutation type (e.g., "tra_peptide_mhc_one") |

**Note:** For old format data without `permutation_key`, individual molecule columns (`tra`, `trb`, `peptide`, etc.) are preserved.

## Loading Tokenized Data

### Python (HuggingFace Datasets)

```python
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk('path/to/output/hf_dataset')

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# View example
print(train_data[0])
# {
#   'input_ids': [0, 5, 12, 3, ...],
#   'attention_mask': [1, 1, 1, 1, ...],
#   'sequence': 'CASSLGQAYEQYF GILGFVFTL',
#   'permutation_key': 'tra_peptide'
# }

# Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('path/to/output/tokenizer')
```

### Python (Pandas)

```python
import pandas as pd

# Load parquet (for compatibility)
df = pd.read_parquet('path/to/output/train/*.parquet')
```

## Sampling Strategies Explained

### No Sampling (Default)
```bash
# Uses entire dataset
--input-dir path/to/data --output-dir path/to/output --model-type esm2
```
- Processes all sequences in the input directory
- Use when you have sufficient compute and want all data

### First-N Sampling
```bash
# Takes first 1M sequences encountered
--sample 1000000
```
- Fast: stops after collecting N sequences
- Distribution depends on file order
- Use for quick testing/prototyping

### Proportional Sampling
```bash
# Maintains original permutation distribution
--sample 1000000 --sample-mode proportional
```
- Analyzes full dataset distribution first
- Samples from each permutation proportionally
- If dataset is 60% TRA, 40% peptide → sample maintains 60/40
- Use when you want a representative subset

### Balanced Sampling
```bash
# Equal samples per permutation type
--sample 1000000 --sample-mode balanced
```
- Analyzes full dataset distribution first
- Divides sample size equally across all permutation types
- 1M samples ÷ 10 permutations = 100k per permutation
- Use when you want equal representation (prevents class imbalance)

## Performance Tuning

### Memory Optimization

**For 600GB RAM machines:**
```bash
--chunk-size 20000000    # 20M rows = ~3GB per chunk
--num-workers 50         # 50 workers × 3GB = ~150GB peak
```

**For 128GB RAM machines:**
```bash
--chunk-size 5000000     # 5M rows = ~750MB per chunk
--num-workers 16         # 16 workers × 750MB = ~12GB peak
```

### Speed Optimization

**Fast tokenization:**
- Increase `--num-workers` (up to CPU count - 2)
- Increase `--chunk-size` if you have RAM headroom
- Use fast storage (NVMe) for input/output directories

**Sampling modes:**
- `first-N`: Fastest (no distribution analysis)
- `proportional`: Medium (analyzes distribution once)
- `balanced`: Medium (analyzes distribution once)

## Common Use Cases

### 1. Training a Foundation Model (Use All Data)
```bash
python scripts/data_processing/tokenize_sequences.py \
  --input-dir data/deduplicated \
  --output-dir data/tokenized/foundation \
  --model-type esm2 \
  --max-length 1024 \
  --num-workers 60
```

### 2. Quick Prototyping (100k Samples)
```bash
python scripts/data_processing/tokenize_sequences.py \
  --input-dir data/deduplicated \
  --output-dir data/tokenized/prototype \
  --model-type protbert \
  --max-length 512 \
  --sample 100000 \
  --num-workers 32
```

### 3. Balanced Multi-Task Learning
```bash
python scripts/data_processing/tokenize_sequences.py \
  --input-dir data/deduplicated \
  --output-dir data/tokenized/multitask \
  --model-type esm2 \
  --max-length 1024 \
  --sample 5000000 \
  --sample-mode balanced \
  --num-workers 50
```

### 4. Fine-tuning Experiment (Proportional Sample)
```bash
python scripts/data_processing/tokenize_sequences.py \
  --input-dir data/deduplicated \
  --output-dir data/tokenized/finetune \
  --model-type protbert \
  --max-length 512 \
  --sample 1000000 \
  --sample-mode proportional \
  --num-workers 40
```

## Next Steps

After tokenization, proceed to pre-masking:

```bash
# See PRE_MASKING_GUIDE.md for details
python scripts/data_processing/pre_mask_dataset.py \
  --input_dataset path/to/output/hf_dataset \
  --output_dataset path/to/masked_output \
  --mode mlm \
  --tokenizer_path Rostlab/prot_bert \
  --num_proc 60
```

## Troubleshooting

### Out of Memory
- Reduce `--chunk-size` to 5M or lower
- Reduce `--num-workers`
- Process on a machine with more RAM

### Slow Tokenization
- Increase `--num-workers` (up to CPU count)
- Use faster storage (NVMe instead of HDD)
- Increase `--chunk-size` if you have RAM headroom

### Unbalanced Dataset
- Use `--sample-mode balanced` for equal representation
- Check distribution with: analyze full dataset first

### Missing Tokenizer
For BPE models:
- First run: add `--train-bpe` flag
- Subsequent runs: remove `--train-bpe` (loads from disk)

## Related Documentation

- [Pre-Masking Guide](PRE_MASKING_GUIDE.md) - Apply task-specific masking
- [Training Guide](TRAINING_GUIDE.md) - Train models with tokenized data
- [Deduplication Guide](DEDUPLICATION_GUIDE.md) - Prepare input data
