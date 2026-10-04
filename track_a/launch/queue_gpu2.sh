#!/usr/bin/env bash
# GPU 2: LSTM × 4 main tasks → MLP × 4 main tasks (smaller models, fast).
set -e
cd /home/ubuntu/quest
LOG=/home/ubuntu/quest/track_a/launch/gpu2.log
exec > >(tee -a "$LOG") 2>&1
echo "[gpu2] started at $(date -u +%FT%TZ)"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate base

for model in lstm mlp; do
  for task in as_trb_i as_paired_i as_trb_ii as_paired_ii; do
    echo "[gpu2] running $model / $task at $(date -u +%FT%TZ)"
    bash track_a/launch/run_baseline.sh "$model" "$task" 2 || { echo "[gpu2] FAILED $model/$task"; break 2; }
  done
done
echo "[gpu2] all cells complete at $(date -u +%FT%TZ)"
