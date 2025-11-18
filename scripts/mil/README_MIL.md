# MIL-Based TCR Repertoire Classification

This directory contains scripts for training Multiple Instance Learning (MIL) models for TCR repertoire classification using embeddings from fine-tuned ESM2 models.

## Overview

The MIL framework treats each repertoire as a "bag" of TCR sequences (instances) and learns to:
1. **Predict** the repertoire-level label (e.g., disease status)
2. **Identify** the top-k most important TCR sequences contributing to the prediction

## Files

### Main Scripts

- **`prepare_mil_data.py`**: Convert TSV files to Parquet format and create JSON datasets
- **`repertoire_mil_adaptive.py`**: Train and evaluate attention-based MIL classifier
- **`run_mil_pipeline.sh`**: End-to-end pipeline script

### Legacy Scripts

- **`repertoire_mil.py`**: Original MIL implementation (kept for reference)

## Quick Start

### 1. Run the Complete Pipeline

The easiest way to get started is to use the pipeline script:

```bash
cd /home/ubuntu/quest
bash scripts/inference/run_mil_pipeline.sh
```

This will:
1. Convert train_dataset_1 TSV files to Parquet format
2. Create JSON format for MIL training
3. Create a sampled dataset (50 repertoires) for quick testing
4. Extract embeddings using the ESM2-650M fine-tuned model
5. Train an attention-based MIL classifier on the sample

### 2. Manual Step-by-Step Workflow

If you prefer to run steps individually:

#### Step 1: Prepare the Data

```bash
python scripts/data_processing/prepare_mil_data.py \
    --input_dir /home/ubuntu/quest/data/mil/train_datasets/train_datasets/train_dataset_1 \
    --output_dir /home/ubuntu/quest/data/mil/processed/train_dataset_1 \
    --convert_to_parquet \
    --create_json \
    --create_sample \
    --n_sample_repertoires 50 \
    --max_sequences_per_repertoire 500
```

This creates:
- `parquet/`: Parquet files (more efficient than TSV)
- `repertoire_data.json`: Full dataset in JSON format
- `repertoire_data_sample.json`: Sampled dataset for testing

#### Step 2: Train MIL Model on Sample Data

```bash
python scripts/inference/repertoire_mil_adaptive.py \
    --data_json /home/ubuntu/quest/data/mil/processed/train_dataset_1/repertoire_data_sample.json \
    --peft_model_path /home/ubuntu/quest/checkpoints/esm2_650M_cdr_dmlm_1M_balanced \
    --embeddings_cache /home/ubuntu/quest/data/mil/processed/train_dataset_1/embeddings_cache_sample.h5 \
    --output_dir /home/ubuntu/quest/results/mil_train_dataset_1_sample \
    --pooling mean \
    --hidden_dim 256 \
    --dropout 0.3 \
    --num_epochs 20 \
    --batch_size 32 \
    --lr 0.0001 \
    --patience 10 \
    --device cuda \
    --use_amp \
    --top_k 100
```

#### Step 3: Train on Full Dataset (Optional)

Once you've validated the approach on the sample:

```bash
python scripts/inference/repertoire_mil_adaptive.py \
    --data_json /home/ubuntu/quest/data/mil/processed/train_dataset_1/repertoire_data.json \
    --peft_model_path /home/ubuntu/quest/checkpoints/esm2_650M_cdr_dmlm_1M_balanced \
    --embeddings_cache /home/ubuntu/quest/data/mil/processed/train_dataset_1/embeddings_cache_full.h5 \
    --output_dir /home/ubuntu/quest/results/mil_train_dataset_1_full \
    --pooling mean \
    --hidden_dim 256 \
    --dropout 0.3 \
    --num_epochs 50 \
    --batch_size 32 \
    --lr 0.0001 \
    --patience 10 \
    --device cuda \
    --use_amp \
    --use_additional_features \
    --top_k 50000
```

## Data Format

### Input Format (TSV)

Each repertoire is stored as a TSV file with columns:
- `junction_aa`: TCR beta CDR3 amino acid sequence
- `v_call`: V gene name
- `j_call`: J gene name
- `d_call`: D gene name (optional)
- `templates`: Frequency count (optional)

Metadata file (`metadata.csv`) contains:
- `repertoire_id`: Unique identifier
- `filename`: TSV filename
- `label_positive`: Binary label (True/False)

### Intermediate Format (JSON)

The JSON format is a list of repertoire dictionaries:

```json
[
  {
    "repertoire_id": "abc123",
    "sequences": ["CASSLGQANTGELFF", "CASSQETQYF", ...],
    "label": 1,
    "v_gene": ["TRBV7-2", "TRBV5-1", ...],
    "j_gene": ["TRBJ2-2", "TRBJ2-5", ...],
    "cdr3_length": [15, 10, ...],
    "frequency": [0.05, 0.03, ...]
  },
  ...
]
```

## Output Files

After training, the output directory contains:

- **`best_model.pt`**: Trained MIL model weights
- **`test_results.json`**: Overall evaluation metrics
  - Accuracy, precision, recall, F1 score, AUC
  - Confusion matrix
- **`repertoire_predictions.json`**: Per-repertoire predictions
  - Predicted and true labels
  - Prediction probabilities
  - Top-k important sequences with attention weights
- **`training_curves.png`**: Training/validation loss and accuracy plots
- **`confusion_matrix.png`**: Confusion matrix heatmap

## Key Features

### Memory Efficiency

1. **HDF5 Embedding Cache**: Embeddings are stored in compressed HDF5 format with lazy loading
2. **Streaming Extraction**: Embeddings are extracted and written incrementally
3. **Gradient Accumulation**: Effective batch size without loading all data at once
4. **Mixed Precision (FP16)**: Reduces memory usage and speeds up training
5. **Parquet Format**: More efficient than TSV for storage and loading

### Additional Features

The model can incorporate additional features beyond sequence embeddings:
- V/J gene usage (encoded as indices)
- Sequence frequency (log-scaled)
- CDR3 length (normalized)

Enable with `--use_additional_features` flag.

### Attention Mechanism

The attention-based MIL model learns to:
- Assign importance weights to each TCR sequence
- Aggregate sequences using learned attention weights
- Identify which sequences are most predictive

The top-k sequences by attention weight are saved for biological interpretation.

## Parameter Tuning

### Key Hyperparameters

- **`--hidden_dim`**: Size of hidden layers (default: 256)
  - Larger values = more model capacity but slower training
  - Try: 128, 256, 512

- **`--dropout`**: Dropout rate (default: 0.3)
  - Higher values = more regularization
  - Try: 0.2, 0.3, 0.5

- **`--lr`**: Learning rate (default: 1e-4)
  - Too high = unstable training
  - Too low = slow convergence
  - Try: 1e-5, 1e-4, 1e-3

- **`--pooling`**: Embedding pooling strategy (default: mean)
  - `mean`: Average over sequence length
  - `cls`: Use [CLS] token embedding
  - `max`: Max pooling over sequence length

- **`--accumulation_steps`**: Gradient accumulation (default: 8)
  - Effective batch size = accumulation_steps × 1
  - Higher values = more stable gradients but slower updates

### Training Tips

1. **Start with sample data**: Validate your approach quickly
2. **Monitor validation metrics**: Use early stopping to prevent overfitting
3. **Check attention weights**: Ensure model is learning meaningful patterns
4. **Try additional features**: V/J genes can provide useful signal
5. **Experiment with pooling**: Different strategies may work better for your data

## Performance Optimization

### GPU Utilization

- Use `--use_amp` for mixed precision (2-3x speedup)
- Adjust `--batch_size` for embedding extraction (larger = faster)
- Set `--num_workers 2-4` for data loading (prefetching)
- Use `--accumulation_steps` to simulate larger batch sizes

### CPU/RAM Efficiency

- Use HDF5 caching (automatic with `--embeddings_cache`)
- Parquet format is more efficient than TSV
- Set `--num_workers 0` if RAM is very limited
- Sample large repertoires with `prepare_mil_data.py`

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce `--batch_size` (for embedding extraction)
2. Increase `--accumulation_steps` (for training)
3. Disable mixed precision: `--no_amp`
4. Use CPU: `--device cpu` (much slower)

### Poor Performance

1. Check label balance (should be relatively balanced)
2. Verify embeddings are cached correctly
3. Try different `--pooling` strategies
4. Increase `--hidden_dim` or model capacity
5. Add `--use_additional_features`
6. Reduce `--dropout` if underfitting

### Slow Training

1. Enable `--use_amp` for mixed precision
2. Increase `--batch_size` for embedding extraction
3. Use `--num_workers 2-4` for prefetching
4. Check GPU utilization with `nvidia-smi`

## Advanced Usage

### Using W&B for Experiment Tracking

```bash
python scripts/inference/repertoire_mil_adaptive.py \
    --data_json ... \
    --wandb_project "tcr-mil-classification" \
    --wandb_run_name "train_dataset_1_attention_256" \
    ...
```

### Custom Train/Val/Test Splits

Adjust split fractions:
```bash
--test_size 0.2    # 20% test
--val_size 0.1     # 10% of train becomes validation
```

### Extracting Top Sequences for All Datasets

The challenge requires top 50,000 sequences per dataset:

```bash
python scripts/inference/repertoire_mil_adaptive.py \
    --top_k 50000 \
    ...
```

Results are saved in `repertoire_predictions.json`.

## Citation

If you use this code, please cite the original MIL paper:

```
Ilse, M., Tomczak, J., & Welling, M. (2018).
Attention-based deep multiple instance learning.
In International conference on machine learning (pp. 2127-2136). PMLR.
```

## Contact

For questions or issues, please refer to the main project README or open an issue.
