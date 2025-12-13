# MIL Multi-Dataset Comparison Pipeline

This pipeline trains and evaluates Multiple Instance Learning (MIL) models across 8 training datasets with global clustering.

## Overview

The pipeline is split into 3 phases for cost optimization:

| Phase | Description | Compute | Est. Time |
|-------|-------------|---------|-----------|
| **Phase 1** | Extract embeddings | GPU | ~2-4 hours per dataset |
| **Phase 2** | Global clustering | CPU | ~30-60 min per dataset |
| **Phase 3** | Train MIL models | GPU | ~10-20 min per model |

By separating CPU-bound clustering (Phase 2), you can run it on cheaper CPU instances.

## Quick Start

```bash
# Run entire pipeline on a single GPU instance
cd /home/ubuntu/quest
./scripts/mil/run_pipeline.sh --all

# Or run phases separately:
./scripts/mil/run_pipeline.sh --phase1  # GPU instance
./scripts/mil/run_pipeline.sh --phase2  # Can be CPU-only instance  
./scripts/mil/run_pipeline.sh --phase3  # GPU instance
```

## Directory Structure

```
data/mil/
├── train_datasets/train_datasets/    # Source data (TSV files)
│   ├── train_dataset_1/
│   ├── train_dataset_2/
│   └── ...
├── processed_mil/                    # Pre-processed repertoire data
│   ├── train_dataset_1/
│   │   ├── unique_sequences.parquet
│   │   ├── repertoires.parquet
│   │   ├── repertoires.json
│   │   └── metadata.parquet
│   └── ...
├── embeddings/                       # Phase 1 outputs
│   ├── train_dataset_1_embeddings.parquet
│   └── ...
├── clustering/                       # Phase 2 outputs
│   ├── train_dataset_1/
│   │   ├── kmeans/
│   │   │   ├── model.pkl
│   │   │   ├── sequence_clusters.parquet
│   │   │   └── centroids.npy
│   │   └── leiden/
│   │       ├── sequence_clusters.parquet
│   │       └── community_info.json
│   └── ...
├── clustered/                        # Phase 2 repertoire assignments
│   ├── train_dataset_1/
│   │   ├── kmeans/
│   │   │   ├── clustered.pkl
│   │   │   └── clustered.json
│   │   └── leiden/
│   │       ├── clustered.pkl
│   │       └── clustered.json
│   └── ...
├── results/                          # Phase 3 training results
│   ├── train_dataset_1/
│   │   ├── kmeans/
│   │   │   ├── hybrid/
│   │   │   │   ├── best_model.pt
│   │   │   │   └── results.json
│   │   │   ├── transmil/
│   │   │   ├── dsmil/
│   │   │   └── abmil/
│   │   └── leiden/
│   └── ...
├── analysis/                         # Aggregated results
│   ├── full_results.csv
│   ├── pivot_auc_method_dataset.csv
│   ├── comparison_plots.png
│   └── statistics.json
└── checkpoints/                      # Pipeline checkpoints
    ├── phase1_checkpoint.json
    ├── phase2_checkpoint.json
    └── phase3_training_checkpoint.json
```

## Phase 1: Embedding Extraction (GPU)

Extracts ProtBERT embeddings for all unique sequences in each dataset.

```bash
./scripts/mil/pipeline_phase1_embeddings.sh [--resume] [--datasets "1 2 3..."]
```

**Key script:** `scripts/mil/data/extract_embeddings_protbert.py`

**Model:** `checkpoints/protbert_cdr_dmlm_1M_balanced/best_model`

**Output:** `data/mil/embeddings/{dataset}_embeddings.parquet`
- ~30GB per dataset with 7M+ sequences
- 1024-dimensional embeddings

## Phase 2: Global Clustering (CPU)

Clusters all sequences globally using K-means (K=100) and Leiden algorithms.

```bash
./scripts/mil/pipeline_phase2_clustering.sh [--resume] [--datasets "1 2 3..."]
```

### Step 2a: Create Global Clusters

**Script:** `scripts/mil/data/create_global_clusters.py`

**Methods:**
- **K-means (K=100):** MiniBatch K-means with sample weights for datasets 7-8
- **Leiden:** Community detection on k-NN graph

**Output:** `data/mil/clustering/{dataset}/{method}/`

### Step 2b: Assign Clusters to Repertoires

**Script:** `scripts/mil/data/assign_repertoire_clusters.py`

Creates per-repertoire cluster representations:
- Cluster centroids as "instances"
- Aggregate weights per cluster
- Both .pkl and .json formats for compatibility

**Output:** `data/mil/clustered/{dataset}/{method}/clustered.pkl`

## Phase 3: MIL Training (GPU)

Trains 4 MIL architectures on clustered data.

```bash
./scripts/mil/pipeline_phase3_mil_training.sh [--resume] [--datasets "1 2..."] [--methods "hybrid transmil..."]
```

**Script:** `scripts/mil/training/train_global_cluster_mil.py`

**Models:**
| Model | Description |
|-------|-------------|
| `hybrid` | Hybrid attention with local + global attention |
| `transmil` | Transformer-based MIL with CLS token |
| `dsmil` | Dual-stream MIL with instance + bag branches |
| `abmil` | Standard attention-based MIL |

**Training Parameters:**
- Epochs: 50 (with early stopping)
- Batch size: 16
- Learning rate: 0.0001
- Train/Val/Test split: 80/10/10
- Patience: 10 epochs

**Output:** `data/mil/results/{dataset}/{clustering}/{method}/results.json`

## Results Aggregation

After Phase 3, aggregate all results:

```bash
python scripts/mil/analysis/aggregate_results.py \
    --results_dir data/mil/results \
    --output_dir data/mil/analysis
```

Generates:
- Summary tables (CSV, LaTeX)
- Comparison plots
- Statistical significance tests

## Dataset Information

| Dataset | Unique Sequences | Repertoires | Notes |
|---------|-----------------|-------------|-------|
| 1-6 | ~7-9M | ~1000 | No count column |
| 7-8 | ~50M | ~1000 | Has `templates` column for frequency weighting |

## Resuming from Failures

All phases support resuming from checkpoints:

```bash
# Resume any phase
./scripts/mil/run_pipeline.sh --phase1 --resume
./scripts/mil/run_pipeline.sh --phase2 --resume
./scripts/mil/run_pipeline.sh --phase3 --resume
```

Checkpoints are stored in `data/mil/checkpoints/`.

## Selective Processing

Process specific datasets or methods:

```bash
# Only datasets 1 and 2
./scripts/mil/run_pipeline.sh --all --datasets "1 2"

# Only hybrid and transmil
./scripts/mil/pipeline_phase3_mil_training.sh --methods "hybrid transmil"

# Only K-means clustering
./scripts/mil/pipeline_phase2_clustering.sh --clustering "kmeans"
```

## Expected Results

After training on all 8 datasets × 2 clustering × 4 models = 64 runs:

```
data/mil/results/
├── train_dataset_1/kmeans/{hybrid,transmil,dsmil,abmil}/results.json
├── train_dataset_1/leiden/{hybrid,transmil,dsmil,abmil}/results.json
├── train_dataset_2/...
└── ...
```

Summary in `data/mil/analysis/`:
- `full_results.csv`: All metrics for all runs
- `pivot_auc_method_dataset.csv`: AUC comparison table
- `comparison_plots.png`: Visual comparison
- `statistics.json`: Statistical significance tests
