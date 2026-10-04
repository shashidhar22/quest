#!/usr/bin/env bash
# GPU 2 resume: ESM-C 300M base + cpt across the headline tasks (TRB-I, Paired-I).
# class II ESM-C cells go to GPU 3.
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/resume_gpu2.log
exec > >(tee -a "$LOG") 2>&1
echo "[resume/gpu2] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate base

for cell in esmc_300m-base esmc_300m-cpt; do
  for task in as_trb_i as_paired_i; do
    bash track_a/launch/resume_run.sh "$cell" "$task" 2 2000 || { echo "[resume/gpu2] FAILED $cell/$task"; break 2; }
  done
done
echo "[resume/gpu2] all cells complete at $(date -u +%FT%TZ)"
