#!/bin/bash
#
# Phase 3: MIL Training Pipeline (GPU Required)
#
# Trains all MIL methods on clustered repertoire data.
# Run this on a GPU instance after Phase 2 (clustering) is complete.
#
# Usage:
#   ./pipeline_phase3_mil_training.sh [--resume] [--datasets "1 2 3..."] [--methods "hybrid transmil..."]
#
# Prerequisites:
#   - Clustered data in data/mil/clustered/train_dataset_{N}/{kmeans,leiden}/clustered.pkl
#   - Run pipeline_phase2_clustering.sh first (on CPU instance)
#
# MIL Methods:
#   1. hybrid     - Hybrid Attention MIL
#   2. transmil   - TransMIL (Transformer-based)
#   3. dsmil      - Dual-Stream MIL
#   4. abmil      - Attention-Based MIL
#   5. multiscale - Multi-Scale MIL
#   6. leiden_mil - Leiden Community MIL (clustering-aware)
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/mil"
OUTPUT_DIR="${DATA_DIR}/results"
CHECKPOINT_FILE="${DATA_DIR}/checkpoints/phase3_training_checkpoint.json"

# Training parameters
NUM_EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=0.0001
TEST_SIZE=0.1        # 10% test
VAL_SIZE=0.111       # 10% of remaining 90% = ~10% validation -> 80/10/10 split
SEED=42

# Parse arguments
RESUME=false
DATASETS="1 2 3 4 5 6 7 8"
METHODS="hybrid transmil dsmil abmil multiscale"
CLUSTERING_METHODS="kmeans leiden"

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME=true
            shift
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --clustering)
            CLUSTERING_METHODS="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$(dirname "${CHECKPOINT_FILE}")"
mkdir -p "${OUTPUT_DIR}"

# Initialize checkpoint file if not exists
if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo '{}' > "${CHECKPOINT_FILE}"
fi

echo "========================================================================"
echo "PHASE 3: MIL TRAINING (GPU)"
echo "========================================================================"
echo "Datasets: ${DATASETS}"
echo "MIL Methods: ${METHODS}"
echo "Clustering: ${CLUSTERING_METHODS}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train/Val/Test split: 80/10/10"
echo "Resume mode: ${RESUME}"
echo ""

# Check GPU availability
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" || {
    echo "ERROR: PyTorch with CUDA not available"
    exit 1
}
echo ""

# Function to check if training is completed
is_training_completed() {
    local dataset=$1
    local clustering=$2
    local method=$3
    python3 -c "
import json
with open('${CHECKPOINT_FILE}') as f:
    cp = json.load(f)
key = 'train_dataset_${dataset}_${clustering}_${method}'
status = cp.get(key, {}).get('status', 'pending')
exit(0 if status == 'completed' else 1)
" 2>/dev/null
}

# Function to mark training as completed
mark_training_completed() {
    local dataset=$1
    local clustering=$2
    local method=$3
    local auc=$4
    python3 -c "
import json
from datetime import datetime
with open('${CHECKPOINT_FILE}') as f:
    cp = json.load(f)
key = 'train_dataset_${dataset}_${clustering}_${method}'
cp[key] = {
    'status': 'completed',
    'timestamp': datetime.now().isoformat(),
    'auc': ${auc}
}
with open('${CHECKPOINT_FILE}', 'w') as f:
    json.dump(cp, f, indent=2)
"
}

# Function to mark training as in progress
mark_training_in_progress() {
    local dataset=$1
    local clustering=$2
    local method=$3
    python3 -c "
import json
from datetime import datetime
with open('${CHECKPOINT_FILE}') as f:
    cp = json.load(f)
key = 'train_dataset_${dataset}_${clustering}_${method}'
cp[key] = {
    'status': 'in_progress',
    'started': datetime.now().isoformat()
}
with open('${CHECKPOINT_FILE}', 'w') as f:
    json.dump(cp, f, indent=2)
"
}

# Training functions for each MIL method
# All use train_global_cluster_mil.py which works with cluster centroids from Phase 2

train_hybrid() {
    local clustered_path=$1
    local output_dir=$2
    local dataset_name=$3
    
    python3 "${SCRIPT_DIR}/training/train_global_cluster_mil.py" \
        --data_path "${clustered_path}" \
        --output_dir "${output_dir}" \
        --model hybrid \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LEARNING_RATE} \
        --test_size ${TEST_SIZE} \
        --val_size ${VAL_SIZE} \
        --seed ${SEED}
}

train_transmil() {
    local clustered_path=$1
    local output_dir=$2
    local dataset_name=$3
    
    python3 "${SCRIPT_DIR}/training/train_global_cluster_mil.py" \
        --data_path "${clustered_path}" \
        --output_dir "${output_dir}" \
        --model transmil \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LEARNING_RATE} \
        --test_size ${TEST_SIZE} \
        --val_size ${VAL_SIZE} \
        --seed ${SEED}
}

train_dsmil() {
    local clustered_path=$1
    local output_dir=$2
    local dataset_name=$3
    
    python3 "${SCRIPT_DIR}/training/train_global_cluster_mil.py" \
        --data_path "${clustered_path}" \
        --output_dir "${output_dir}" \
        --model dsmil \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LEARNING_RATE} \
        --test_size ${TEST_SIZE} \
        --val_size ${VAL_SIZE} \
        --seed ${SEED}
}

train_abmil() {
    local clustered_path=$1
    local output_dir=$2
    local dataset_name=$3
    
    python3 "${SCRIPT_DIR}/training/train_global_cluster_mil.py" \
        --data_path "${clustered_path}" \
        --output_dir "${output_dir}" \
        --model abmil \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LEARNING_RATE} \
        --test_size ${TEST_SIZE} \
        --val_size ${VAL_SIZE} \
        --seed ${SEED}
}

# Multiscale uses different approach - per-repertoire multi-resolution clustering
# For now, skip or use hybrid as fallback
train_multiscale() {
    local clustered_path=$1
    local output_dir=$2
    local dataset_name=$3
    
    echo "  Note: Multi-scale MIL uses per-repertoire clustering, using hybrid for global clusters"
    python3 "${SCRIPT_DIR}/training/train_global_cluster_mil.py" \
        --data_path "${clustered_path}" \
        --output_dir "${output_dir}" \
        --model hybrid \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LEARNING_RATE} \
        --test_size ${TEST_SIZE} \
        --val_size ${VAL_SIZE} \
        --seed ${SEED}
}

# Process each combination
TOTAL_RUNS=0
COMPLETED_RUNS=0
FAILED_RUNS=0

for DATASET_NUM in ${DATASETS}; do
    DATASET_NAME="train_dataset_${DATASET_NUM}"
    
    for CLUSTERING in ${CLUSTERING_METHODS}; do
        CLUSTERED_PATH="${DATA_DIR}/clustered/${DATASET_NAME}/${CLUSTERING}/clustered.pkl"
        
        # Check if clustered data exists
        if [ ! -f "${CLUSTERED_PATH}" ]; then
            echo "Warning: Clustered data not found: ${CLUSTERED_PATH}"
            echo "  Skipping ${DATASET_NAME} with ${CLUSTERING} clustering - run Phase 2 first"
            continue
        fi
        
        for MIL_METHOD in ${METHODS}; do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
            RUN_NAME="${DATASET_NAME}_${CLUSTERING}_${MIL_METHOD}"
            RUN_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET_NAME}/${CLUSTERING}/${MIL_METHOD}"
            
            echo ""
            echo "========================================================================"
            echo "Training: ${RUN_NAME}"
            echo "========================================================================"
            echo "  Data: ${CLUSTERED_PATH}"
            echo "  Output: ${RUN_OUTPUT_DIR}"
            
            # Check if already completed
            if [ "${RESUME}" = true ] && is_training_completed "${DATASET_NUM}" "${CLUSTERING}" "${MIL_METHOD}"; then
                echo "  ✓ Already completed, skipping"
                COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
                continue
            fi
            
            # Mark as in progress
            mark_training_in_progress "${DATASET_NUM}" "${CLUSTERING}" "${MIL_METHOD}"
            mkdir -p "${RUN_OUTPUT_DIR}"
            
            START_TIME=$(date +%s)
            
            # Run training based on method
            case ${MIL_METHOD} in
                hybrid)
                    train_hybrid "${CLUSTERED_PATH}" "${RUN_OUTPUT_DIR}" "${DATASET_NAME}"
                    ;;
                transmil)
                    train_transmil "${CLUSTERED_PATH}" "${RUN_OUTPUT_DIR}" "${DATASET_NAME}"
                    ;;
                dsmil)
                    train_dsmil "${CLUSTERED_PATH}" "${RUN_OUTPUT_DIR}" "${DATASET_NAME}"
                    ;;
                abmil)
                    train_abmil "${CLUSTERED_PATH}" "${RUN_OUTPUT_DIR}" "${DATASET_NAME}"
                    ;;
                multiscale)
                    train_multiscale "${CLUSTERED_PATH}" "${RUN_OUTPUT_DIR}" "${DATASET_NAME}"
                    ;;
                *)
                    echo "  Unknown method: ${MIL_METHOD}, skipping"
                    continue
                    ;;
            esac
            
            EXIT_CODE=$?
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            
            if [ ${EXIT_CODE} -eq 0 ]; then
                # Extract AUC from results if available
                if [ -f "${RUN_OUTPUT_DIR}/results.json" ]; then
                    AUC=$(python3 -c "import json; print(json.load(open('${RUN_OUTPUT_DIR}/results.json')).get('test_auc', 0.0))" 2>/dev/null || echo "0.0")
                else
                    AUC="0.0"
                fi
                
                mark_training_completed "${DATASET_NUM}" "${CLUSTERING}" "${MIL_METHOD}" "${AUC}"
                COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
                echo "  ✓ Completed in ${DURATION}s (AUC: ${AUC})"
            else
                FAILED_RUNS=$((FAILED_RUNS + 1))
                echo "  ✗ Failed after ${DURATION}s"
            fi
        done
    done
done

echo ""
echo "========================================================================"
echo "PHASE 3 COMPLETE"
echo "========================================================================"
echo ""
echo "Results: ${OUTPUT_DIR}/"
echo "Total runs: ${TOTAL_RUNS}"
echo "Completed: ${COMPLETED_RUNS}"
echo "Failed: ${FAILED_RUNS}"
echo ""
echo "Checkpoint file: ${CHECKPOINT_FILE}"
echo ""
