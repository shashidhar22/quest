#!/bin/bash
#
# Phase 2: Global Clustering Pipeline (CPU-Optimized)
#
# Performs global K-means and Leiden clustering on all datasets.
# Optimized for large CPU instances (e.g., x2gd.16xlarge: 64 vCPUs, 512GB RAM)
#
# Usage:
#   ./pipeline_phase2_clustering.sh [--resume] [--datasets "1 2 3..."] [--method kmeans|leiden|both]
#
# Prerequisites:
#   - Embeddings must exist in data/mil/embeddings/train_dataset_{N}_embeddings.parquet
#   - Run pipeline_phase1_embeddings.sh first (on GPU instance)
#
# Output:
#   data/mil/global_clusters/train_dataset_{1-8}/{kmeans,leiden}/
#   data/mil/clustered/train_dataset_{1-8}/{kmeans,leiden}/clustered.pkl
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/mil"
CHECKPOINT_FILE="${DATA_DIR}/checkpoints/phase2_clustering_checkpoint.json"

# Clustering parameters
N_CLUSTERS=100              # K-means clusters
LEIDEN_RESOLUTION=1.0       # Leiden resolution
LEIDEN_NEIGHBORS=15         # k-NN neighbors for Leiden
KMEANS_BATCH_SIZE=10000     # MiniBatchKMeans batch size

# Detect CPU cores
N_JOBS=$(nproc)

# Parse arguments
RESUME=false
DATASETS="1 2 3 4 5 6 7 8"
METHOD="both"

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
        --method)
            METHOD="$2"
            shift 2
            ;;
        --n_clusters)
            N_CLUSTERS="$2"
            shift 2
            ;;
        --resolution)
            LEIDEN_RESOLUTION="$2"
            shift 2
            ;;
        --n_jobs)
            N_JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create checkpoint directory
mkdir -p "$(dirname "${CHECKPOINT_FILE}")"

# Initialize checkpoint file if not exists
if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo '{}' > "${CHECKPOINT_FILE}"
fi

echo "========================================================================"
echo "PHASE 2: GLOBAL CLUSTERING (CPU)"
echo "========================================================================"
echo "Datasets: ${DATASETS}"
echo "Method: ${METHOD}"
echo "K-means clusters: ${N_CLUSTERS}"
echo "Leiden resolution: ${LEIDEN_RESOLUTION}"
echo "CPU cores: ${N_JOBS}"
echo "Resume mode: ${RESUME}"
echo ""

# Check memory
MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "Available memory: ${MEM_GB} GB"

if [ "${MEM_GB}" -lt 64 ]; then
    echo "Warning: Less than 64GB RAM. Large datasets (7, 8) may fail."
    echo "Consider using a larger instance for datasets 7-8."
fi
echo ""

# Function to check if step is completed
is_step_completed() {
    local dataset=$1
    local method=$2
    python3 -c "
import json
with open('${CHECKPOINT_FILE}') as f:
    cp = json.load(f)
status = cp.get('train_dataset_${dataset}', {}).get('${method}', {}).get('status', 'pending')
exit(0 if status == 'completed' else 1)
" 2>/dev/null
}

# Function to mark step as completed
mark_step_completed() {
    local dataset=$1
    local method=$2
    local n_clusters=$3
    python3 -c "
import json
from datetime import datetime
with open('${CHECKPOINT_FILE}') as f:
    cp = json.load(f)
if 'train_dataset_${dataset}' not in cp:
    cp['train_dataset_${dataset}'] = {}
cp['train_dataset_${dataset}']['${method}'] = {
    'status': 'completed',
    'timestamp': datetime.now().isoformat(),
    'n_clusters': ${n_clusters}
}
with open('${CHECKPOINT_FILE}', 'w') as f:
    json.dump(cp, f, indent=2)
"
}

# Function to mark step as in progress
mark_step_in_progress() {
    local dataset=$1
    local method=$2
    python3 -c "
import json
from datetime import datetime
with open('${CHECKPOINT_FILE}') as f:
    cp = json.load(f)
if 'train_dataset_${dataset}' not in cp:
    cp['train_dataset_${dataset}'] = {}
cp['train_dataset_${dataset}']['${method}'] = {
    'status': 'in_progress',
    'started': datetime.now().isoformat()
}
with open('${CHECKPOINT_FILE}', 'w') as f:
    json.dump(cp, f, indent=2)
"
}

# Process each dataset
for DATASET_NUM in ${DATASETS}; do
    DATASET_NAME="train_dataset_${DATASET_NUM}"
    EMBEDDINGS_FILE="${DATA_DIR}/embeddings/${DATASET_NAME}_embeddings.parquet"
    DATASET_TSV_DIR="${DATA_DIR}/train_datasets/train_datasets/${DATASET_NAME}"
    PROCESSED_DIR="${DATA_DIR}/processed_mil/${DATASET_NAME}"
    GLOBAL_CLUSTERS_DIR="${DATA_DIR}/global_clusters/${DATASET_NAME}"
    CLUSTERED_DIR="${DATA_DIR}/clustered/${DATASET_NAME}"
    
    echo ""
    echo "========================================================================"
    echo "Processing: ${DATASET_NAME}"
    echo "========================================================================"
    
    # Check if embeddings exist
    if [ ! -f "${EMBEDDINGS_FILE}" ]; then
        echo "  ERROR: Embeddings not found: ${EMBEDDINGS_FILE}"
        echo "  Run Phase 1 (embedding extraction) first."
        continue
    fi
    
    # Check embedding file size
    EMBEDDING_SIZE=$(du -h "${EMBEDDINGS_FILE}" | cut -f1)
    echo "  Embeddings: ${EMBEDDINGS_FILE} (${EMBEDDING_SIZE})"
    
    # ========================================================================
    # Step 1: Global Clustering
    # ========================================================================
    
    RUN_KMEANS=false
    RUN_LEIDEN=false
    
    if [ "${METHOD}" = "kmeans" ] || [ "${METHOD}" = "both" ]; then
        if [ "${RESUME}" = true ] && is_step_completed "${DATASET_NUM}" "kmeans"; then
            echo "  ✓ K-means clustering already completed, skipping"
        else
            RUN_KMEANS=true
        fi
    fi
    
    if [ "${METHOD}" = "leiden" ] || [ "${METHOD}" = "both" ]; then
        if [ "${RESUME}" = true ] && is_step_completed "${DATASET_NUM}" "leiden"; then
            echo "  ✓ Leiden clustering already completed, skipping"
        else
            RUN_LEIDEN=true
        fi
    fi
    
    # Determine which method to run
    if [ "${RUN_KMEANS}" = true ] && [ "${RUN_LEIDEN}" = true ]; then
        CLUSTER_METHOD="both"
    elif [ "${RUN_KMEANS}" = true ]; then
        CLUSTER_METHOD="kmeans"
    elif [ "${RUN_LEIDEN}" = true ]; then
        CLUSTER_METHOD="leiden"
    else
        echo "  All clustering steps completed for ${DATASET_NAME}"
        
        # Still need to check if cluster assignment is done
        # Fall through to assignment step
        CLUSTER_METHOD="none"
    fi
    
    if [ "${CLUSTER_METHOD}" != "none" ]; then
        echo ""
        echo "  Running global clustering (${CLUSTER_METHOD})..."
        
        if [ "${RUN_KMEANS}" = true ]; then
            mark_step_in_progress "${DATASET_NUM}" "kmeans"
        fi
        if [ "${RUN_LEIDEN}" = true ]; then
            mark_step_in_progress "${DATASET_NUM}" "leiden"
        fi
        
        START_TIME=$(date +%s)
        
        python3 "${SCRIPT_DIR}/data/create_global_clusters.py" \
            --embeddings_path "${EMBEDDINGS_FILE}" \
            --output_dir "${GLOBAL_CLUSTERS_DIR}" \
            --dataset_dir "${DATASET_TSV_DIR}" \
            --dataset_num "${DATASET_NUM}" \
            --method "${CLUSTER_METHOD}" \
            --n_clusters "${N_CLUSTERS}" \
            --kmeans_batch_size "${KMEANS_BATCH_SIZE}" \
            --resolution "${LEIDEN_RESOLUTION}" \
            --n_neighbors "${LEIDEN_NEIGHBORS}" \
            --n_jobs "${N_JOBS}" \
            --seed 42
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "  Clustering completed in ${DURATION}s"
        
        # Mark steps as completed
        if [ "${RUN_KMEANS}" = true ]; then
            mark_step_completed "${DATASET_NUM}" "kmeans" "${N_CLUSTERS}"
        fi
        if [ "${RUN_LEIDEN}" = true ]; then
            # Get actual cluster count from metrics file
            if [ -f "${GLOBAL_CLUSTERS_DIR}/leiden_metrics.json" ]; then
                LEIDEN_CLUSTERS=$(python3 -c "import json; print(json.load(open('${GLOBAL_CLUSTERS_DIR}/leiden_metrics.json'))['n_clusters'])")
            else
                LEIDEN_CLUSTERS=0
            fi
            mark_step_completed "${DATASET_NUM}" "leiden" "${LEIDEN_CLUSTERS}"
        fi
    fi
    
    # ========================================================================
    # Step 2: Assign Clusters to Repertoires
    # ========================================================================
    
    echo ""
    echo "  Assigning clusters to repertoires..."
    
    # K-means assignment
    if [ "${METHOD}" = "kmeans" ] || [ "${METHOD}" = "both" ]; then
        KMEANS_CLUSTERED="${CLUSTERED_DIR}/kmeans/clustered.pkl"
        
        if [ "${RESUME}" = true ] && [ -f "${KMEANS_CLUSTERED}" ]; then
            echo "    ✓ K-means cluster assignment already exists, skipping"
        else
            echo "    Assigning K-means clusters..."
            mkdir -p "$(dirname "${KMEANS_CLUSTERED}")"
            
            python3 "${SCRIPT_DIR}/data/assign_repertoire_clusters.py" \
                --clustering_dir "${GLOBAL_CLUSTERS_DIR}" \
                --processed_dir "${PROCESSED_DIR}" \
                --output_path "${KMEANS_CLUSTERED}" \
                --method kmeans \
                --use_counts
            
            echo "    ✓ Saved: ${KMEANS_CLUSTERED}"
        fi
    fi
    
    # Leiden assignment
    if [ "${METHOD}" = "leiden" ] || [ "${METHOD}" = "both" ]; then
        LEIDEN_CLUSTERED="${CLUSTERED_DIR}/leiden/clustered.pkl"
        
        if [ "${RESUME}" = true ] && [ -f "${LEIDEN_CLUSTERED}" ]; then
            echo "    ✓ Leiden cluster assignment already exists, skipping"
        else
            echo "    Assigning Leiden clusters..."
            mkdir -p "$(dirname "${LEIDEN_CLUSTERED}")"
            
            python3 "${SCRIPT_DIR}/data/assign_repertoire_clusters.py" \
                --clustering_dir "${GLOBAL_CLUSTERS_DIR}" \
                --processed_dir "${PROCESSED_DIR}" \
                --output_path "${LEIDEN_CLUSTERED}" \
                --method leiden \
                --use_counts
            
            echo "    ✓ Saved: ${LEIDEN_CLUSTERED}"
        fi
    fi
    
    echo ""
    echo "  ✓ ${DATASET_NAME} clustering complete"
done

echo ""
echo "========================================================================"
echo "PHASE 2 COMPLETE"
echo "========================================================================"
echo ""
echo "Global clustering results: ${DATA_DIR}/global_clusters/"
echo "Repertoire clusters: ${DATA_DIR}/clustered/"
echo "Checkpoint file: ${CHECKPOINT_FILE}"
echo ""
echo "Next step: Sync clustered.pkl files to shared storage, then run Phase 3 on GPU instance"
echo ""
