# Tokenization Summary

## Overview
Created tokenization pipeline supporting 5 different model architectures. All tokenizers preserve original sequences and individual molecule data.

## Supported Models

### 1. **ProtBERT**
- **Type**: BERT-based protein language model
- **Tokenization**: Space-separated amino acids
- **Pre-trained**: `Rostlab/prot_bert`
- **Vocab Size**: 30 (standard BERT vocab)
- **Special Tokens**: [CLS], [SEP], [PAD], [MASK], [UNK]

### 2. **BERT** 
- **Type**: Standard BERT (same as ProtBERT)
- **Tokenization**: Space-separated amino acids
- **Use Case**: Alternative to ProtBERT for comparison

### 3. **ESM-2** (Evolutionary Scale Modeling)
- **Type**: Facebook's evolutionary protein model
- **Tokenization**: Direct amino acid encoding (no spaces)
- **Pre-trained**: `facebook/esm2_t6_8M_UR50D`
- **Vocab Size**: 33 (ESM vocabulary)
- **Special Tokens**: <cls>, <eos>, <pad>, <mask>, <unk>

### 4. **ESM-3**
- **Type**: Next-gen ESM model
- **Tokenization**: Similar to ESM-2
- **Note**: Falls back to ESM-2 if not available

### 5. **BPE (Byte Pair Encoding)**
- **Type**: Custom tokenizer for LSTM/Transformer
- **Tokenization**: Learned subword units
- **Vocab Size**: Configurable (default 2000)
- **Special Tokens**: Custom molecule boundaries
  - `[TRA]`, `[ETRA]` - TRA chain boundaries
  - `[TRB]`, `[ETRB]` - TRB chain boundaries
  - `[PEP]`, `[EPEP]` - Peptide boundaries
  - `[MHC1]`, `[EMHC1]` - MHC class I boundaries
  - `[MHC2]`, `[EMHC2]` - MHC class II boundaries
- **Training Required**: Yes (--train-bpe flag)

## Output Format

### HuggingFace Dataset
Location: `{output_dir}/hf_dataset/`

**Features:**
- `input_ids` - Tokenized sequence IDs (List[int])
- `attention_mask` - Attention mask (List[int])
- `sequence` - Original concatenated sequence (str)
- `tra` - TRA chain sequence (str)
- `trb` - TRB chain sequence (str)  
- `peptide` - Peptide sequence (str)
- `mhc_one` - MHC class I sequence (str)
- `mhc_two` - MHC class II sequence (str)

**Splits:**
- `train` (80%)
- `validation` (10%)
- `test` (10%)

### Parquet Files (Compatibility)
Location: `{output_dir}/{split}/{split}.parquet`

Same schema as HuggingFace dataset but in Parquet format for compatibility with other tools.

### Tokenizer
Location: `{output_dir}/tokenizer/`

Saved tokenizer that can be loaded for inference.

## Usage

### Basic Tokenization
```bash
python scripts/data_processing/tokenize_sequences.py \
    --input-dir /path/to/deduplicated \
    --output-dir /path/to/output \
    --model-type protbert \
    --max-length 512
```

### BPE Training + Tokenization
```bash
python scripts/data_processing/tokenize_sequences.py \
    --input-dir /path/to/deduplicated \
    --output-dir /path/to/output \
    --model-type bpe \
    --max-length 512 \
    --vocab-size 2000 \
    --train-bpe
```

### With Sampling (for testing)
```bash
python scripts/data_processing/tokenize_sequences.py \
    --input-dir /path/to/deduplicated \
    --output-dir /path/to/output \
    --model-type esm2 \
    --max-length 256 \
    --sample 10000
```

## Loading Tokenized Data

### Python
```python
from datasets import load_from_disk

# Load HuggingFace dataset
dataset = load_from_disk('/path/to/output/hf_dataset')

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Access examples
example = train_data[0]
print(example['sequence'])
print(example['input_ids'])
print(example['attention_mask'])
```

### Pandas
```python
import pandas as pd

# Load parquet
df = pd.read_parquet('/path/to/output/train/train.parquet')
```


