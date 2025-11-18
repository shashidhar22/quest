#!/bin/bash
#
# Full MIL Pipeline for TCR Repertoire Classification
# =====================================================
#
# This script runs the complete pipeline:
# 1. Data preparation (Parquet format)
# 2. Global embedding extraction (multi-GPU)
# 3. Per-dataset clustering
# 4. MIL training
# 5. Top TCR extraction
#
# Usage:
#   ./run_full_pipeline.sh [stage]
#
# Stages:
#   prepare   - Data preparation only
#   embed     - Embedding extraction only (run on multi-GPU instance)
#   cluster   - Clustering only
#   train     - Training only
#   extract   - Top TCR extraction only
#   all       - Run complete pipeline (default)

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
# These can be set via environment variables or modified here

# Required paths (no defaults - must be set)
DATA_BASE="${DATA_BASE:-}"
MODEL_PATH="${MODEL_PATH:-}"

# Derived paths (can be overridden)
PROCESSED_BASE="${PROCESSED_BASE:-${DATA_BASE}/processed_mil}"
RESULTS_BASE="${RESULTS_BASE:-${DATA_BASE}/../results/mil}"

# Training datasets (space-separated list, can be overridden)
if [ -z "${TRAIN_DATASETS_STR:-}" ]; then
    TRAIN_DATASETS=(
        "train_dataset_1"
        "train_dataset_2"
        "train_dataset_3"
        "train_dataset_4"
        "train_dataset_5"
        "train_dataset_6"
        "train_dataset_7"
        "train_dataset_8"
    )
else
    IFS=' ' read -ra TRAIN_DATASETS <<< "${TRAIN_DATASETS_STR}"
fi

# Clustering parameters
N_CLUSTERS="${N_CLUSTERS:-100}"
CLUSTERING_METHOD="${CLUSTERING_METHOD:-kmeans}"

# Training parameters
NUM_EPOCHS="${NUM_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
PATIENCE="${PATIENCE:-15}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"

# Extraction parameters
TOP_K="${TOP_K:-50000}"

# ============================================================================
# VALIDATION
# ============================================================================
validate_config() {
    local errors=0

    if [ -z "${DATA_BASE}" ]; then
        echo "ERROR: DATA_BASE is not set"
        echo "  Set via: export DATA_BASE=/path/to/mil/data"
        errors=$((errors + 1))
    fi

    if [ -z "${MODEL_PATH}" ]; then
        echo "ERROR: MODEL_PATH is not set"
        echo "  Set via: export MODEL_PATH=/path/to/model"
        errors=$((errors + 1))
    fi

    if [ ${errors} -gt 0 ]; then
        echo ""
        echo "Required environment variables:"
        echo "  DATA_BASE     - Base directory containing train_datasets/"
        echo "  MODEL_PATH    - Path to fine-tuned ESM2 model"
        echo ""
        echo "Optional environment variables:"
        echo "  PROCESSED_BASE    - Output for processed data (default: \${DATA_BASE}/processed_mil)"
        echo "  RESULTS_BASE      - Output for results (default: \${DATA_BASE}/../results/mil)"
        echo "  TRAIN_DATASETS_STR - Space-separated dataset names (default: train_dataset_1-8)"
        echo "  N_CLUSTERS        - Number of clusters (default: 100)"
        echo "  NUM_EPOCHS        - Training epochs (default: 100)"
        echo "  BATCH_SIZE        - Training batch size (default: 16)"
        echo "  LR                - Learning rate (default: 1e-3)"
        echo "  PATIENCE          - Early stopping patience (default: 15)"
        echo "  HIDDEN_DIM        - Hidden layer dimension (default: 256)"
        echo "  TOP_K             - Top TCRs to extract (default: 50000)"
        exit 1
    fi
}

# ============================================================================
# STAGE 1: DATA PREPARATION
# ============================================================================
prepare_data() {
    echo "============================================================================"
    echo "STAGE 1: DATA PREPARATION"
    echo "============================================================================"

    for dataset in "${TRAIN_DATASETS[@]}"; do
        INPUT_DIR="${DATA_BASE}/train_datasets/train_datasets/${dataset}"
        OUTPUT_DIR="${PROCESSED_BASE}/${dataset}"

        if [ ! -d "${INPUT_DIR}" ]; then
            echo "Skipping ${dataset}: not found at ${INPUT_DIR}"
            continue
        fi

        if [ -f "${OUTPUT_DIR}/repertoires.parquet" ]; then
            echo "Skipping ${dataset}: already prepared"
            continue
        fi

        echo ""
        echo "Processing ${dataset}..."
        python scripts/mil/data/prepare_mil_parquet.py \
            --input_dir "${INPUT_DIR}" \
            --output_dir "${OUTPUT_DIR}"
    done

    # Merge all unique sequences
    echo ""
    echo "Merging unique sequences across all datasets..."
    PROCESSED_DIRS=()
    for dataset in "${TRAIN_DATASETS[@]}"; do
        PROCESSED_DIRS+=("${PROCESSED_BASE}/${dataset}")
    done

    python scripts/mil/data/prepare_mil_parquet.py \
        --merge_datasets ${PROCESSED_DIRS[@]} \
        --merge_output "${PROCESSED_BASE}/global_unique_sequences.parquet"

    echo ""
    echo "Data preparation complete!"
}

# ============================================================================
# STAGE 2: EMBEDDING EXTRACTION (Multi-GPU)
# ============================================================================
extract_embeddings() {
    echo "============================================================================"
    echo "STAGE 2: EMBEDDING EXTRACTION (Multi-GPU)"
    echo "============================================================================"

    GLOBAL_SEQS="${PROCESSED_BASE}/global_unique_sequences.parquet"
    GLOBAL_EMBS="${PROCESSED_BASE}/global_embeddings.parquet"

    if [ ! -f "${GLOBAL_SEQS}" ]; then
        echo "Error: Global unique sequences not found at ${GLOBAL_SEQS}"
        echo "Run: ./run_full_pipeline.sh prepare"
        exit 1
    fi

    if [ -f "${GLOBAL_EMBS}" ]; then
        echo "Embeddings already exist at ${GLOBAL_EMBS}"
        echo "Delete to re-extract."
        return
    fi

    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected ${NUM_GPUS} GPUs"

    python scripts/mil/data/extract_embeddings_multi_gpu.py \
        --sequences_parquet "${GLOBAL_SEQS}" \
        --output_parquet "${GLOBAL_EMBS}" \
        --model_path "${MODEL_PATH}" \
        --batch_size 512 \
        --num_workers 4 \
        --fp16 \
        --chunk_size 1000000

    echo ""
    echo "Embedding extraction complete!"
}

# ============================================================================
# STAGE 3: CLUSTERING
# ============================================================================
cluster_data() {
    echo "============================================================================"
    echo "STAGE 3: CLUSTERING"
    echo "============================================================================"

    GLOBAL_EMBS="${PROCESSED_BASE}/global_embeddings.parquet"

    if [ ! -f "${GLOBAL_EMBS}" ]; then
        echo "Error: Global embeddings not found at ${GLOBAL_EMBS}"
        echo "Run: ./run_full_pipeline.sh embed"
        exit 1
    fi

    for dataset in "${TRAIN_DATASETS[@]}"; do
        REPERTOIRES="${PROCESSED_BASE}/${dataset}/repertoires.parquet"
        OUTPUT_PKL="${RESULTS_BASE}/${dataset}/clustered_${CLUSTERING_METHOD}_k${N_CLUSTERS}.pkl"

        if [ ! -f "${REPERTOIRES}" ]; then
            echo "Skipping ${dataset}: repertoires.parquet not found"
            continue
        fi

        if [ -f "${OUTPUT_PKL}" ]; then
            echo "Skipping ${dataset}: already clustered"
            continue
        fi

        mkdir -p "$(dirname ${OUTPUT_PKL})"

        echo ""
        echo "Clustering ${dataset}..."
        python scripts/mil/data/create_clustered_parquet.py \
            --repertoires_parquet "${REPERTOIRES}" \
            --embeddings_parquet "${GLOBAL_EMBS}" \
            --output_pkl "${OUTPUT_PKL}" \
            --method "${CLUSTERING_METHOD}" \
            --n_clusters ${N_CLUSTERS}
    done

    echo ""
    echo "Clustering complete!"
}

# ============================================================================
# STAGE 4: MIL TRAINING
# ============================================================================
train_models() {
    echo "============================================================================"
    echo "STAGE 4: MIL TRAINING"
    echo "============================================================================"

    for dataset in "${TRAIN_DATASETS[@]}"; do
        CLUSTERED="${RESULTS_BASE}/${dataset}/clustered_${CLUSTERING_METHOD}_k${N_CLUSTERS}.pkl"
        OUTPUT_DIR="${RESULTS_BASE}/${dataset}/mil_model"

        if [ ! -f "${CLUSTERED}" ]; then
            echo "Skipping ${dataset}: clustered data not found"
            continue
        fi

        if [ -f "${OUTPUT_DIR}/best_model.pt" ]; then
            echo "Skipping ${dataset}: model already trained"
            continue
        fi

        mkdir -p "${OUTPUT_DIR}"

        echo ""
        echo "Training MIL model for ${dataset}..."
        python scripts/mil/training/repertoire_mil_clustering.py \
            --precomputed_clusters "${CLUSTERED}" \
            --output_dir "${OUTPUT_DIR}" \
            --num_epochs ${NUM_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --patience ${PATIENCE} \
            --hidden_dims ${HIDDEN_DIM} \
            --dropout 0.3 \
            --device cuda
    done

    echo ""
    echo "Training complete!"
}

# ============================================================================
# STAGE 5: TOP TCR EXTRACTION
# ============================================================================
extract_tcrs() {
    echo "============================================================================"
    echo "STAGE 5: TOP TCR EXTRACTION"
    echo "============================================================================"

    for dataset in "${TRAIN_DATASETS[@]}"; do
        MODEL="${RESULTS_BASE}/${dataset}/mil_model/best_model.pt"
        CLUSTERED="${RESULTS_BASE}/${dataset}/clustered_${CLUSTERING_METHOD}_k${N_CLUSTERS}.pkl"
        REPERTOIRES="${PROCESSED_BASE}/${dataset}/repertoires.parquet"
        OUTPUT_CSV="${RESULTS_BASE}/${dataset}/top_${TOP_K}_tcrs.csv"

        if [ ! -f "${MODEL}" ]; then
            echo "Skipping ${dataset}: model not found"
            continue
        fi

        if [ -f "${OUTPUT_CSV}" ]; then
            echo "Skipping ${dataset}: TCRs already extracted"
            continue
        fi

        echo ""
        echo "Extracting top TCRs for ${dataset}..."
        python scripts/mil/analysis/extract_top_tcrs.py \
            --model_path "${MODEL}" \
            --clustered_pkl "${CLUSTERED}" \
            --repertoires_parquet "${REPERTOIRES}" \
            --output_csv "${OUTPUT_CSV}" \
            --top_k ${TOP_K} \
            --hidden_dim ${HIDDEN_DIM}
    done

    echo ""
    echo "TCR extraction complete!"
}

# ============================================================================
# MAIN
# ============================================================================
main() {
    STAGE=${1:-all}

    # Validate configuration
    validate_config

    echo "============================================================================"
    echo "MIL PIPELINE - Stage: ${STAGE}"
    echo "============================================================================"
    echo ""
    echo "Configuration:"
    echo "  Data base: ${DATA_BASE}"
    echo "  Processed: ${PROCESSED_BASE}"
    echo "  Results: ${RESULTS_BASE}"
    echo "  Model: ${MODEL_PATH}"
    echo "  Clusters: ${N_CLUSTERS} (${CLUSTERING_METHOD})"
    echo "  Datasets: ${TRAIN_DATASETS[*]}"
    echo "  Training: epochs=${NUM_EPOCHS}, batch=${BATCH_SIZE}, lr=${LR}"
    echo ""

    case ${STAGE} in
        prepare)
            prepare_data
            ;;
        embed)
            extract_embeddings
            ;;
        cluster)
            cluster_data
            ;;
        train)
            train_models
            ;;
        extract)
            extract_tcrs
            ;;
        all)
            prepare_data
            extract_embeddings
            cluster_data
            train_models
            extract_tcrs
            ;;
        *)
            echo "Unknown stage: ${STAGE}"
            echo "Valid stages: prepare, embed, cluster, train, extract, all"
            exit 1
            ;;
    esac

    echo ""
    echo "============================================================================"
    echo "PIPELINE COMPLETE"
    echo "============================================================================"
}

main "$@"
