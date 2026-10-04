#!/usr/bin/env bash
# GPU 3 resume: TCR-BERT remaining + ESM-C 300M class II cells.
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/resume_gpu3.log
exec > >(tee -a "$LOG") 2>&1
echo "[resume/gpu3] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate base

# 1. TCR-BERT remaining (TRB-II has checkpoint already; just eval)
for task in as_trb_ii as_paired_ii; do
  bash track_a/launch/resume_run.sh tcrbert "$task" 3 || echo "[resume/gpu3] tcrbert/$task FAILED, continuing"
done

# 2. ESM-C 300M class II cells (eval-on-class-II using class-I-trained checkpoint).
#    Note: trains on class-I companion data, so similar wall-clock to class I cells.
for cell in esmc_300m-base esmc_300m-cpt; do
  for task in as_trb_ii as_paired_ii; do
    bash track_a/launch/resume_run.sh "$cell" "$task" 3 2000 || echo "[resume/gpu3] $cell/$task FAILED, continuing"
  done
done
echo "[resume/gpu3] all cells complete at $(date -u +%FT%TZ)"
