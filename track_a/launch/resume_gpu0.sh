#!/usr/bin/env bash
# GPU 0 resume: finish ESM2-8M base + cpt cells.
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/resume_gpu0.log
exec > >(tee -a "$LOG") 2>&1
echo "[resume/gpu0] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate base

for cell in esm2_8m-base esm2_8m-cpt; do
  for task in as_trb_i as_paired_i as_trb_ii as_paired_ii; do
    bash track_a/launch/resume_run.sh "$cell" "$task" 0 || { echo "[resume/gpu0] FAILED $cell/$task"; break 2; }
  done
done
echo "[resume/gpu0] all cells complete at $(date -u +%FT%TZ)"
