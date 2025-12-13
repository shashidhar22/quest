#!/bin/bash
#
# Phase 1: Embedding Extraction Pipeline (GPU Required)
#
# Extracts ProtBERT embeddings for all unique TCR sequences across 8 training datasets.
# Run this on a GPU instance, then sync embeddings to shared storage.
#
# Usage:
#   ./pipeline_phase1_embeddings.sh [--resume] [--datasets "1 2 3..."]
#
# Output:
#   data/mil/embeddings/train_dataset_{1-8}_embeddings.parquet
#
# Estimated time: ~2-4 hours per dataset depending on GPU and sequence count
# Estimated storage: ~30GB per dataset for 1024-dim embeddings
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/mil"
CHECKPOINT_FILE="${DATA_DIR}/checkpoints/phase1_embeddings_checkpoint.json"

# Model configuration
MODEL_PATH="${PROJECT_ROOT}/checkpoints/protbert_cdr_dmlm_1M_balanced/best_model"
BATCH_SIZE=256  # Adjust based on GPU memory
NUM_WORKERS=4
MAX_LENGTH=100
POOLING="mean"
CHUNK_SIZE=500000

# Parse arguments
RESUME=false
DATASETS="1 2 3 4 5 6 7 8"

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
        --model)
            MODEL_PATH="$2"
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

# Create checkpoint directory
mkdir -p "$(dirname "${CHECKPOINT_FILE}")"

# Initialize checkpoint file if not exists
if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo '{}' > "${CHECKPOINT_FILE}"
fi

echo "========================================================================"
echo "PHASE 1: EMBEDDING EXTRACTION (GPU)"
echo "========================================================================"
echo "Model: ${MODEL_PATH}"
echo "Datasets: ${DATASETS}"
echo "Resume mode: ${RESUME}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

# Check GPU availability
python3 -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')" || {
    echo "ERROR: PyTorch with CUDA not available"
    exit 1
}

# Function to check if dataset embedding is complete
is_completed() {
    local dataset=$1
    python3 -c "
import json
with open('${CHECKPOINT_FILE}') as f:
    cp = json.load(f)
status = cp.get('train_dataset_${dataset}', {}).get('status', 'pending')
exit(0 if status == 'completed' else 1)
" 2>/dev/null
}

# Function to mark dataset as completed
mark_completed() {
    local dataset=$1
    local num_sequences=$2
    python3 -c "
import json
from datetime import datetime
with open('${CHECKPOINT_FILE}') as f:
    cp = json.load(f)
cp['train_dataset_${dataset}'] = {
    'status': 'completed',
    'timestamp': datetime.now().isoformat(),
    'num_sequences': ${num_sequences}
}
with open('${CHECKPOINT_FILE}', 'w') as f:
    json.dump(cp, f, indent=2)
"
}

# Function to mark dataset as in progress
mark_in_progress() {
    local dataset=$1
    python3 -c "
import json
from datetime import datetime
with open('${CHECKPOINT_FILE}') as f:
    cp = json.load(f)
cp['train_dataset_${dataset}'] = {
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
    SEQUENCES_FILE="${DATA_DIR}/processed_mil/${DATASET_NAME}/unique_sequences.parquet"
    OUTPUT_FILE="${DATA_DIR}/embeddings/${DATASET_NAME}_embeddings.parquet"
    
    echo ""
    echo "========================================================================"
    echo "Processing: ${DATASET_NAME}"
    echo "========================================================================"
    
    # Check if already completed
    if [ "${RESUME}" = true ] && is_completed "${DATASET_NUM}"; then
        echo "  ✓ Already completed, skipping (use --no-resume to force)"
        continue
    fi
    
    # Check if input exists
    if [ ! -f "${SEQUENCES_FILE}" ]; then
        echo "  ERROR: Sequences file not found: ${SEQUENCES_FILE}"
        echo "  Run preprocessing first to generate unique_sequences.parquet"
        continue
    fi
    
    # Check sequence count
    NUM_SEQUENCES=$(python3 -c "
import pyarrow.parquet as pq
pf = pq.ParquetFile('${SEQUENCES_FILE}')
print(pf.metadata.num_rows)
")
    echo "  Sequences: ${NUM_SEQUENCES}"
    
    # Mark as in progress
    mark_in_progress "${DATASET_NUM}"
    
    # Run embedding extraction
    echo "  Extracting embeddings..."
    START_TIME=$(date +%s)
    
    python3 "${SCRIPT_DIR}/data/extract_embeddings_protbert.py" \
        --sequences_parquet "${SEQUENCES_FILE}" \
        --output_parquet "${OUTPUT_FILE}" \
        --model_path "${MODEL_PATH}" \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS} \
        --max_length ${MAX_LENGTH} \
        --pooling ${POOLING} \
        --chunk_size ${CHUNK_SIZE} \
        --fp16
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # Verify output
    if [ -f "${OUTPUT_FILE}" ]; then
        FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
        echo "  ✓ Completed in ${DURATION}s"
        echo "  Output: ${OUTPUT_FILE} (${FILE_SIZE})"
        
        # Mark as completed
        mark_completed "${DATASET_NUM}" "${NUM_SEQUENCES}"
    else
        echo "  ✗ FAILED: Output file not created"
        exit 1
    fi
done

echo ""
echo "========================================================================"
echo "PHASE 1 COMPLETE"
echo "========================================================================"
echo ""
echo "Embedding files created in: ${DATA_DIR}/embeddings/"
ls -lh "${DATA_DIR}/embeddings/"*.parquet 2>/dev/null || echo "No embedding files found"
echo ""
echo "Checkpoint file: ${CHECKPOINT_FILE}"
echo ""
echo "Next step: Sync embeddings to shared storage (S3/EFS), then run Phase 2 on CPU instance"
echo ""
