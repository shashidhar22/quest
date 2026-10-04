#!/usr/bin/env bash
# GPU 1 resume: finish ESM2-35M base + cpt cells.
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/resume_gpu1.log
exec > >(tee -a "$LOG") 2>&1
echo "[resume/gpu1] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate base

for cell in esm2_35m-base esm2_35m-cpt; do
  for task in as_trb_i as_paired_i as_trb_ii as_paired_ii; do
    bash track_a/launch/resume_run.sh "$cell" "$task" 1 || { echo "[resume/gpu1] FAILED $cell/$task"; break 2; }
  done
done
echo "[resume/gpu1] all cells complete at $(date -u +%FT%TZ)"
