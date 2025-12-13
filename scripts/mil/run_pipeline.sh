#!/bin/bash
#
# MIL Pipeline Master Orchestrator
#
# This script coordinates all three phases of the MIL training pipeline:
#   Phase 1: Extract embeddings (GPU)
#   Phase 2: Cluster sequences (CPU)  
#   Phase 3: Train MIL models (GPU)
#
# For cost optimization, Phase 2 can be run on a cheaper CPU-only instance.
#
# Usage:
#   # Run all phases on a single GPU instance:
#   ./run_pipeline.sh --all
#
#   # Run individual phases:
#   ./run_pipeline.sh --phase1  # Extract embeddings
#   ./run_pipeline.sh --phase2  # Cluster (can run on CPU instance)
#   ./run_pipeline.sh --phase3  # Train MIL models
#
#   # Resume from failures:
#   ./run_pipeline.sh --all --resume
#
# Options:
#   --datasets "1 2 3..."   - Specify which datasets to process (default: all 8)
#   --methods "hybrid..."   - Specify MIL methods for Phase 3
#   --clustering "kmeans.." - Specify clustering methods for Phase 2 & 3
#   --resume                - Resume from last checkpoint
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default options
RUN_PHASE1=false
RUN_PHASE2=false
RUN_PHASE3=false
RESUME=""
DATASETS=""
METHODS=""
CLUSTERING=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_PHASE1=true
            RUN_PHASE2=true
            RUN_PHASE3=true
            shift
            ;;
        --phase1)
            RUN_PHASE1=true
            shift
            ;;
        --phase2)
            RUN_PHASE2=true
            shift
            ;;
        --phase3)
            RUN_PHASE3=true
            shift
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --datasets)
            DATASETS="--datasets \"$2\""
            shift 2
            ;;
        --methods)
            METHODS="--methods \"$2\""
            shift 2
            ;;
        --clustering)
            CLUSTERING="--clustering \"$2\""
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--all|--phase1|--phase2|--phase3] [--resume] [--datasets N] [--methods M] [--clustering C]"
            echo ""
            echo "Phases:"
            echo "  --phase1   Run embedding extraction (GPU required)"
            echo "  --phase2   Run clustering (CPU, can run on cheaper instance)"
            echo "  --phase3   Run MIL training (GPU required)"
            echo "  --all      Run all phases sequentially"
            echo ""
            echo "Options:"
            echo "  --resume        Resume from last checkpoint"
            echo "  --datasets      Space-separated dataset numbers (e.g., \"1 2 3\")"
            echo "  --methods       MIL methods for phase 3 (e.g., \"hybrid transmil dsmil\")"
            echo "  --clustering    Clustering methods (e.g., \"kmeans leiden\")"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if any phase selected
if ! $RUN_PHASE1 && ! $RUN_PHASE2 && ! $RUN_PHASE3; then
    echo "Error: No phase selected. Use --phase1, --phase2, --phase3, or --all"
    echo "Run with -h for help"
    exit 1
fi

echo "========================================================================"
echo "MIL PIPELINE ORCHESTRATOR"
echo "========================================================================"
echo ""
echo "Phases to run:"
$RUN_PHASE1 && echo "  ✓ Phase 1: Embedding Extraction (GPU)"
$RUN_PHASE2 && echo "  ✓ Phase 2: Clustering (CPU)"
$RUN_PHASE3 && echo "  ✓ Phase 3: MIL Training (GPU)"
echo ""

TOTAL_START=$(date +%s)

# Phase 1: Embeddings
if $RUN_PHASE1; then
    echo ""
    echo "========================================================================"
    echo "PHASE 1: EMBEDDING EXTRACTION"
    echo "========================================================================"
    
    PHASE1_START=$(date +%s)
    
    eval "${SCRIPT_DIR}/pipeline_phase1_embeddings.sh" ${RESUME} ${DATASETS}
    
    PHASE1_END=$(date +%s)
    PHASE1_DURATION=$((PHASE1_END - PHASE1_START))
    echo ""
    echo "Phase 1 completed in ${PHASE1_DURATION} seconds"
fi

# Phase 2: Clustering
if $RUN_PHASE2; then
    echo ""
    echo "========================================================================"
    echo "PHASE 2: CLUSTERING"
    echo "========================================================================"
    
    PHASE2_START=$(date +%s)
    
    eval "${SCRIPT_DIR}/pipeline_phase2_clustering.sh" ${RESUME} ${DATASETS} ${CLUSTERING}
    
    PHASE2_END=$(date +%s)
    PHASE2_DURATION=$((PHASE2_END - PHASE2_START))
    echo ""
    echo "Phase 2 completed in ${PHASE2_DURATION} seconds"
fi

# Phase 3: MIL Training
if $RUN_PHASE3; then
    echo ""
    echo "========================================================================"
    echo "PHASE 3: MIL TRAINING"
    echo "========================================================================"
    
    PHASE3_START=$(date +%s)
    
    eval "${SCRIPT_DIR}/pipeline_phase3_mil_training.sh" ${RESUME} ${DATASETS} ${METHODS} ${CLUSTERING}
    
    PHASE3_END=$(date +%s)
    PHASE3_DURATION=$((PHASE3_END - PHASE3_START))
    echo ""
    echo "Phase 3 completed in ${PHASE3_DURATION} seconds"
fi

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Total time: ${TOTAL_DURATION} seconds ($((TOTAL_DURATION / 3600))h $(((TOTAL_DURATION % 3600) / 60))m $((TOTAL_DURATION % 60))s)"
echo ""
echo "Next steps:"
echo "  1. Aggregate results:"
echo "     python scripts/mil/analysis/aggregate_results.py \\"
echo "       --results_dir data/mil/results \\"
echo "       --output_dir data/mil/analysis"
echo ""
echo "  2. View results in: data/mil/analysis/"
echo ""
