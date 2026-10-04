#!/usr/bin/env bash
# GPU 0: ESM2-8M base + cpt across all 4 main tasks (TRB-I first for headline column).
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/gpu0.log
exec > >(tee -a "$LOG") 2>&1
echo "[gpu0] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate base

for cell in esm2_8m-base esm2_8m-cpt; do
  for task in as_trb_i as_paired_i as_trb_ii as_paired_ii; do
    echo "[gpu0] running $cell / $task at $(date -u +%FT%TZ)"
    bash track_a/launch/run_baseline.sh "$cell" "$task" 0 || { echo "[gpu0] FAILED $cell/$task"; break 2; }
  done
done
echo "[gpu0] all cells complete at $(date -u +%FT%TZ)"
