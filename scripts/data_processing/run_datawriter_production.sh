#!/bin/bash
# Correct usage of ray_datawriter_optimized.py
# 
# This script shows the correct command for processing your 144GB parsed output

# CORRECT COMMAND - Use --path to point to parquet files
python scripts/data_processing/ray_datawriter_optimized.py \
    --path "/mnt/ephemeral/parsed_output/seq/*.parquet" \
    --model-name protbert \
    --mode default \
    --tmp-dir /mnt/ephemeral/temp_dir/ \
    --output-raw /mnt/ephemeral/dataset/mlm_protbert \
    --fast-mode \
    --use-hash-dedup

# EXPLANATION:
# --path: Points to the parquet files to process (the parsed output from run_parser_production.py)
# --input_raw_dir: Is for reading ALREADY TOKENIZED datasets (not what you want)
#
# Your parsed output is in: /mnt/ephemeral/parsed_output/seq/
# This contains 26,563 .parquet files that need to be deduplicated and tokenized

# FOR TESTING (sample 100 files first):
python scripts/data_processing/ray_datawriter_optimized.py \
    --path "/mnt/ephemeral/parsed_output/seq/*.parquet" \
    --model-name protbert \
    --mode default \
    --tmp-dir /mnt/ephemeral/temp_dir/ \
    --output-raw /mnt/ephemeral/dataset/mlm_protbert_test \
    --fast-mode \
    --use-hash-dedup \
    --sample 100

# ALSO: Set environment variable to handle the /dev/shm warning:
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# Then run the command
