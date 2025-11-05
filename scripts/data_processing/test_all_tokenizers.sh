#!/bin/bash
# Test all tokenizers on sample data

SAMPLE_SIZE=1000
INPUT_DIR="/dev/shm/test_dedup_streaming"
BASE_OUTPUT="/dev/shm/tokenized_models"
MAX_LEN=256

echo "Testing all tokenizers on ${SAMPLE_SIZE} samples"
echo "================================================"

# 1. ProtBERT
echo -e "\n1. Testing ProtBERT..."
python scripts/data_processing/tokenize_sequences.py \
    --input-dir ${INPUT_DIR} \
    --output-dir ${BASE_OUTPUT}/protbert \
    --model-type protbert \
    --sample ${SAMPLE_SIZE} \
    --max-length ${MAX_LEN}

# 2. BERT
echo -e "\n2. Testing BERT..."
python scripts/data_processing/tokenize_sequences.py \
    --input-dir ${INPUT_DIR} \
    --output-dir ${BASE_OUTPUT}/bert \
    --model-type bert \
    --sample ${SAMPLE_SIZE} \
    --max-length ${MAX_LEN}

# 3. ESM-2
echo -e "\n3. Testing ESM-2..."
python scripts/data_processing/tokenize_sequences.py \
    --input-dir ${INPUT_DIR} \
    --output-dir ${BASE_OUTPUT}/esm2 \
    --model-type esm2 \
    --sample ${SAMPLE_SIZE} \
    --max-length ${MAX_LEN}

# 4. BPE (train first, then tokenize)
echo -e "\n4. Testing BPE (LSTM/Transformer)..."
python scripts/data_processing/tokenize_sequences.py \
    --input-dir ${INPUT_DIR} \
    --output-dir ${BASE_OUTPUT}/bpe \
    --model-type bpe \
    --sample ${SAMPLE_SIZE} \
    --max-length ${MAX_LEN} \
    --vocab-size 2000 \
    --train-bpe

echo -e "\n================================================"
echo "All tokenizers tested successfully!"
echo "Output directory: ${BASE_OUTPUT}"
