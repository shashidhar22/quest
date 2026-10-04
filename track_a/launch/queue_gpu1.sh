#!/usr/bin/env bash
# GPU 1: ESM2-35M base + cpt across all 4 main tasks.
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/gpu1.log
exec > >(tee -a "$LOG") 2>&1
echo "[gpu1] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate base

for cell in esm2_35m-base esm2_35m-cpt; do
  for task in as_trb_i as_paired_i as_trb_ii as_paired_ii; do
    echo "[gpu1] running $cell / $task at $(date -u +%FT%TZ)"
    bash track_a/launch/run_baseline.sh "$cell" "$task" 1 || { echo "[gpu1] FAILED $cell/$task"; break 2; }
  done
done
echo "[gpu1] all cells complete at $(date -u +%FT%TZ)"
